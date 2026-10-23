use futures::{StreamExt, poll, stream};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use suprnova_ai::{Error, agents::runtime::Runtime, types::Event};

fn chunk(event: Event) -> String {
    match event {
        Event::TextDelta(event) => event.chunk,
        _ => panic!("expected text delta"),
    }
}

#[tokio::test]
async fn source_advances_only_when_requested() {
    let advanced = Arc::new(AtomicUsize::new(0));
    let count = advanced.clone();
    let source = stream::iter(0..10).map(move |i| {
        count.fetch_add(1, Ordering::SeqCst);
        Ok(Event::text_delta(i.to_string()))
    });
    let (_runtime, events) = Runtime::new();
    let mut merged = events.merge(source);
    for i in 0..5 {
        assert_eq!(chunk(merged.next().await.unwrap().unwrap()), i.to_string());
        for _ in 0..50 {
            tokio::task::yield_now().await;
        }
        assert_eq!(advanced.load(Ordering::SeqCst), i + 1);
    }
}

#[tokio::test]
async fn queue_precedes_source_result_from_same_poll() {
    let (runtime, events) = Runtime::new();
    let source = stream::once(async move {
        runtime
            .put_event(Event::text_delta("partial1"))
            .await
            .unwrap();
        runtime
            .put_event(Event::text_delta("partial2"))
            .await
            .unwrap();
        Ok(Event::text_delta("final"))
    });
    let result: Vec<_> = events
        .merge(source)
        .map(|event| chunk(event.unwrap()))
        .collect()
        .await;
    assert_eq!(result, ["partial1", "partial2", "final"]);
}

#[tokio::test]
async fn queue_precedes_source_error_and_drains_at_end() {
    let (runtime, events) = Runtime::new();
    let source = stream::once(async move {
        runtime
            .put_event(Event::text_delta("partial"))
            .await
            .unwrap();
        Err(Error::Validation("source failure".into()))
    });
    let mut merged = events.merge(source);
    assert_eq!(chunk(merged.next().await.unwrap().unwrap()), "partial");
    assert!(matches!(
        merged.next().await,
        Some(Err(Error::Validation(_)))
    ));
    assert!(merged.next().await.is_none());
}

#[tokio::test]
async fn source_completion_closes_queue_with_surviving_producer() {
    let (runtime, events) = Runtime::new();
    runtime.put_event(Event::text_delta("side")).await.unwrap();
    let mut merged = events.merge(stream::empty());
    assert_eq!(chunk(merged.next().await.unwrap().unwrap()), "side");
    assert!(merged.next().await.is_none());
    assert!(runtime.put_event(Event::text_delta("late")).await.is_err());
}

#[tokio::test]
async fn signal_done_is_idempotent_and_drains_before_end() {
    let (runtime, mut events) = Runtime::new();
    runtime
        .put_event(Event::text_delta("queued"))
        .await
        .unwrap();
    runtime.clone().signal_done().await;
    runtime.signal_done().await;
    assert_eq!(chunk(events.next().await.unwrap()), "queued");
    assert!(events.next().await.is_none());
    assert!(runtime.put_event(Event::text_delta("late")).await.is_err());
}

#[tokio::test]
async fn closing_side_channel_does_not_finish_pending_source() {
    let (runtime, events) = Runtime::new();
    let (sender, receiver) = tokio::sync::oneshot::channel();
    let source = stream::once(async move {
        receiver.await.unwrap();
        Ok(Event::text_delta("final"))
    });
    runtime.signal_done().await;
    let mut merged = events.merge(source);
    assert!(poll!(merged.next()).is_pending());
    sender.send(()).unwrap();
    assert_eq!(chunk(merged.next().await.unwrap().unwrap()), "final");
    assert!(merged.next().await.is_none());
}

#[tokio::test]
async fn consumer_drop_closes_producer() {
    let (runtime, events) = Runtime::new();
    for i in 0..1000 {
        runtime
            .put_event(Event::text_delta(i.to_string()))
            .await
            .unwrap();
    }
    drop(events);
    assert!(runtime.put_event(Event::text_delta("late")).await.is_err());
}

struct DropSignal(Arc<AtomicBool>);
impl Drop for DropSignal {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

#[tokio::test]
async fn merged_drop_drops_source_and_closes_sink() {
    let dropped = Arc::new(AtomicBool::new(false));
    let guard = DropSignal(dropped.clone());
    let source = stream::once(async move {
        let _guard = guard;
        futures::future::pending::<Result<Event, Error>>().await
    });
    let (runtime, events) = Runtime::new();
    let mut merged = events.merge(source);
    assert!(poll!(merged.next()).is_pending());
    drop(merged);
    assert!(dropped.load(Ordering::SeqCst));
    assert!(runtime.put_event(Event::text_delta("late")).await.is_err());
}

#[tokio::test]
async fn one_source_poll_can_queue_more_than_sixty_four_partials() {
    let (runtime, events) = Runtime::new();
    let source = stream::once(async move {
        for i in 0..1000 {
            runtime
                .put_event(Event::text_delta(i.to_string()))
                .await
                .unwrap();
        }
        Ok(Event::text_delta("final"))
    });
    let result: Vec<_> = events
        .merge(source)
        .map(|event| chunk(event.unwrap()))
        .collect()
        .await;
    assert_eq!(result.len(), 1001);
    assert_eq!(
        &result[..1000],
        &(0..1000).map(|i| i.to_string()).collect::<Vec<_>>()
    );
    assert_eq!(result[1000], "final");
}
