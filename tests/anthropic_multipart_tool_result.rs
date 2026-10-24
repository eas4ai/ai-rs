mod anthropic_support;
use anthropic_support::*;
use serde_json::{Value, json};
use suprnova_ai::{StreamOptions, types::messages::*};
async fn result_content(result: ToolResultValue, model_input: Option<ToolResultValue>) -> Value {
    let mut part = ToolResultPart::new("tc-1", "read", result);
    if let Some(v) = model_input {
        part.set_model_input(v);
    }
    let (b, _) = capture(
        vec![
            suprnova_ai::user_message("Read image"),
            Message::new(
                MessageRole::Assistant,
                vec![ToolCallPart::new("tc-1", "read", "{}").into()],
            ),
            Message::new(MessageRole::Tool, vec![part.into()]),
        ],
        StreamOptions::default(),
    )
    .await;
    b["messages"][2]["content"][0]["content"].clone()
}
#[tokio::test]
async fn strings_null_objects_lists_and_model_input() {
    for (v, expected) in [
        (json!("hello"), "hello"),
        (Value::Null, ""),
        (json!({"key":"value"}), "{\"key\":\"value\"}"),
        (json!([1, 2, 3]), "[1,2,3]"),
        (json!("boom"), "boom"),
    ] {
        assert_eq!(result_content(v.into(), None).await, expected);
    }
    assert_eq!(
        result_content(json!("private").into(), Some(Value::Null.into())).await,
        ""
    );
}
#[tokio::test]
async fn multipart_text_images_bytes_documents_and_urls() {
    let files = [
        FilePart::from_bytes(b"\x89PNG".to_vec(), Some("image/png"), None).unwrap(),
        FilePart::from_bytes(b"%PDF".to_vec(), Some("application/pdf"), None).unwrap(),
        FilePart::from_bytes(b"plain text".to_vec(), Some("text/plain"), None).unwrap(),
        FilePart::from_url("https://example.com/a.jpg", Some("image/*")).unwrap(),
    ];
    for file in files {
        let mt = file.media_type.clone();
        let data = file.data.clone();
        let content = ContentOutput::new(vec![
            ContentPart::Text(TextPart::new("Image loaded")),
            ContentPart::File(file),
        ]);
        let blocks = result_content(content.into(), None).await;
        assert_eq!(blocks[0], json!({"type":"text","text":"Image loaded"}));
        // A text file cannot go in a tool result; it is left out (ANT-010).
        if mt == "text/plain" {
            assert_eq!(blocks.as_array().unwrap().len(), 1, "{blocks}");
            continue;
        }
        assert_eq!(
            blocks[1]["type"],
            if mt.starts_with("image/") {
                "image"
            } else {
                "document"
            }
        );
        if let FileData::Text(url) = data {
            assert_eq!(blocks[1]["source"]["url"], url);
        }
    }
}
#[tokio::test]
async fn base64_text_document_and_unsupported_media() {
    let text: FilePart =
        serde_json::from_value(json!({"data":"aGVsbG8=","media_type":"text/plain"})).unwrap();
    let (b, _) = capture(
        vec![Message::new(MessageRole::User, vec![text.into()])],
        StreamOptions::default(),
    )
    .await;
    assert_eq!(b["messages"][0]["content"][0]["source"]["data"], "hello");
    let fake = ScriptedTransport::new(vec![]);
    let m = model(fake);
    let file = FilePart::from_bytes(vec![1], Some("audio/wav"), None).unwrap();
    use futures::StreamExt;
    assert!(
        m.provider
            .stream(
                &m,
                vec![Message::new(MessageRole::User, vec![file.into()])],
                StreamOptions::default()
            )
            .next()
            .await
            .unwrap()
            .is_err()
    );
}
