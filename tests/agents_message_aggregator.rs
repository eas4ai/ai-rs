use suprnova_ai::agents::MessageAggregator;
use suprnova_ai::types::events::{
    Aggregator, Event, ModelEventMeta, Retry, StreamEnd, ToolCallResult,
};
use suprnova_ai::types::messages::{Message, MessageRole, TextPart};

fn text(id: &str, value: &str) -> Event {
    Event::StreamEnd(StreamEnd {
        event: ModelEventMeta::with_message(Message::with_id(
            id,
            MessageRole::Assistant,
            vec![TextPart::new(value).into()],
        )),
        ..Default::default()
    })
}

#[test]
fn interleaved_snapshots_replace_without_reordering() {
    let mut aggregator = MessageAggregator::default();
    aggregator.feed(text("a", "a1"));
    let tool = Message::with_id("tool", MessageRole::Tool, vec![]);
    aggregator.feed(Event::ToolCallResult(ToolCallResult {
        replay: false,
        message: tool,
        results: vec![],
        exception: None,
    }));
    aggregator.feed(text("b", "b1"));
    aggregator.feed(text("a", "a2"));
    aggregator.feed(text("b", "b2"));
    let bundle = aggregator.snapshot();
    assert_eq!(
        bundle
            .messages
            .iter()
            .map(|m| m.id.as_str())
            .collect::<Vec<_>>(),
        ["a", "tool", "b"]
    );
    assert_eq!(bundle.messages[0].text(), "a2");
    assert_eq!(aggregator.get_model_input(), "b2");
}

#[test]
fn retry_drops_response_and_subsequent_tool_messages() {
    let mut aggregator = MessageAggregator::default();
    aggregator.feed(Event::Retry(Retry::default()));
    aggregator.feed(text("a", "done"));
    let event = |id| {
        Event::ToolCallResult(ToolCallResult {
            replay: false,
            message: Message::with_id(id, MessageRole::Tool, vec![]),
            results: vec![],
            exception: None,
        })
    };
    aggregator.feed(event("tool-a"));
    aggregator.feed(text("b", "partial"));
    aggregator.feed(event("tool-b"));
    aggregator.feed(Event::Retry(Retry::default()));
    assert_eq!(
        aggregator
            .snapshot()
            .messages
            .iter()
            .map(|m| m.id.as_str())
            .collect::<Vec<_>>(),
        ["a", "tool-a"]
    );
    aggregator.feed(text("c", "retried"));
    aggregator.feed(text("a", "again"));
    let bundle = aggregator.snapshot();
    assert_eq!(
        bundle
            .messages
            .iter()
            .map(|m| m.id.as_str())
            .collect::<Vec<_>>(),
        ["a", "tool-a", "c"]
    );
    assert_eq!(bundle.messages[0].text(), "again");
}

#[test]
fn model_input_uses_last_nonempty_assistant_text() {
    let mut aggregator = MessageAggregator::default();
    aggregator.feed(text("a", "answer"));
    aggregator.feed(text("b", ""));
    assert_eq!(aggregator.get_model_input(), "answer");
    assert_eq!(MessageAggregator::default().get_model_input(), "");
}
