mod anthropic_support;
use anthropic_support::*;
use serde_json::json;
use std::sync::Arc;
use suprnova_ai::{Model, ProviderConfig, providers::anthropic::AnthropicCompatibleProvider};
#[tokio::test]
async fn success_model_not_found_and_custom_version() {
    for status in [200, 404] {
        let fake = ScriptedTransport::new(vec![Ok(response(
            status,
            json!({"id":"custom-model","type":"model"}),
        ))]);
        let p = provider(fake.clone())
            .with_name("custom-anthropic")
            .with_anthropic_version("2024-01-01");
        let m = Model::new("custom/model?x", Arc::new(p));
        let result = m.provider.probe(&m).await;
        if status == 200 {
            result.unwrap();
        } else {
            let suprnova_ai::Error::Provider(err) = result.unwrap_err() else {
                panic!()
            };
            assert_eq!(
                err.kind,
                suprnova_ai::errors::ProviderErrorKind::ModelNotFound
            );
            assert_eq!(err.model_id.as_deref(), Some("custom/model?x"));
        }
        assert_eq!(
            fake.request().url,
            "https://anthropic.test/v1/models/custom%2Fmodel%3Fx"
        );
        assert_eq!(
            header(&fake.request(), "anthropic-version").as_deref(),
            Some("2024-01-01")
        );
    }
}
#[tokio::test]
async fn unconfigured_probe_never_sends_http() {
    let fake = ScriptedTransport::new(vec![]);
    let p = AnthropicCompatibleProvider::new(ProviderConfig::default(), fake.clone());
    let m = Model::new("claude", Arc::new(p));
    let suprnova_ai::Error::Provider(err) = m.provider.probe(&m).await.unwrap_err() else {
        panic!()
    };
    assert_eq!(
        err.kind,
        suprnova_ai::errors::ProviderErrorKind::NotConfigured
    );
    assert!(fake.requests.lock().unwrap().is_empty());
}
