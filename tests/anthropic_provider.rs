// Copyright 2026 Vercel, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

mod anthropic_support;
use anthropic_support::*;
use serde_json::json;
use std::sync::Arc;
use suprnova_ai::{
    Model, Provider, ProviderConfig, ProviderRegistry,
    providers::{
        ModelsDevProvider,
        anthropic::{AnthropicCompatibleProvider, register},
    },
};
#[tokio::test]
async fn list_models_gets_headers_and_sorts_ids() {
    let fake = ScriptedTransport::new(vec![Ok(response(
        200,
        json!({"data":[{"id":"claude-z"},{"id":"claude-a"}]}),
    ))]);
    let p = AnthropicCompatibleProvider::new(
        ProviderConfig {
            default_base_url: "https://anthropic.test".into(),
            api_key_value: Some("sk-test".into()),
            headers: [("X-Custom-Header".into(), "example".into())].into(),
            ..Default::default()
        },
        fake.clone(),
    );
    assert_eq!(p.list_models().await.unwrap(), ["claude-a", "claude-z"]);
    let r = fake.request();
    assert_eq!(r.url, "https://anthropic.test/v1/models");
    assert_eq!(r.method, "GET");
    assert_eq!(header(&r, "x-api-key").as_deref(), Some("sk-test"));
    assert_eq!(
        header(&r, "anthropic-version").as_deref(),
        Some("2023-06-01")
    );
    assert_eq!(header(&r, "x-custom-header").as_deref(), Some("example"));
}
#[test]
fn explicit_config_snapshot_defaults_and_transport_ownership() {
    let fake = ScriptedTransport::new(vec![]);
    let p = AnthropicCompatibleProvider::new(ProviderConfig::default(), fake.clone());
    assert_eq!(p.base_url(), "https://api.anthropic.com");
    assert!(!p.is_configured());
    let p = AnthropicCompatibleProvider::new(
        ProviderConfig {
            env: [
                (
                    "ANTHROPIC_BASE_URL".into(),
                    "https://proxy.example.com".into(),
                ),
                ("ANTHROPIC_API_KEY".into(), "sk-test".into()),
            ]
            .into(),
            ..Default::default()
        },
        fake.clone(),
    );
    assert_eq!(p.base_url(), "https://proxy.example.com");
    assert_eq!(p.api_key(), Some("sk-test"));
    assert!(p.is_configured());
    assert_eq!(
        p.protocol().unwrap().protocol_class_id(),
        "anthropic_messages"
    );
    assert!(p.shared_transport().is_some());
    assert!(!format!("{p:?}").contains("sk-test"));
    let p = AnthropicCompatibleProvider::new(
        ProviderConfig {
            default_base_url: "https://explicit.example.com".into(),
            env: [(
                "ANTHROPIC_BASE_URL".into(),
                "https://ignored.example.com".into(),
            )]
            .into(),
            api_key_value: Some("explicit".into()),
            ..Default::default()
        },
        fake,
    );
    assert_eq!(p.base_url(), "https://explicit.example.com");
}
#[tokio::test]
async fn list_errors_preserve_context_and_404_is_generic() {
    for (status, kind) in [
        (529, suprnova_ai::errors::ProviderErrorKind::Overloaded),
        (404, suprnova_ai::errors::ProviderErrorKind::NotFound),
    ] {
        let fake = ScriptedTransport::new(vec![Ok(response(
            status,
            json!({"error":{"message":"overloaded","type":"overloaded_error","code":42,"param":"model"}}),
        ))]);
        let err = provider(fake).list_models().await.unwrap_err();
        let suprnova_ai::Error::Provider(err) = err else {
            panic!("wrong error")
        };
        assert_eq!(err.kind, kind);
        assert_eq!(err.provider.as_deref(), Some("anthropic"));
        assert_eq!(err.request_id.as_deref(), Some("req-anthropic"));
        assert_eq!(err.error_type.as_deref(), Some("overloaded_error"));
        assert_eq!(err.code.as_deref(), Some("42"));
        assert_eq!(err.param.as_deref(), Some("model"));
        let ctx = err.http_context.unwrap();
        assert_eq!(ctx.status_code, status);
        assert_eq!(ctx.request_method.as_deref(), Some("GET"));
        assert_eq!(
            ctx.request_url.as_deref(),
            Some("https://anthropic.test/v1/models")
        );
    }
}
#[test]
fn registration_restores_model_and_custom_version() {
    let fake = ScriptedTransport::new(vec![]);
    let mut registry = ProviderRegistry::new();
    register(
        &mut registry,
        ProviderConfig {
            api_key_value: Some("sk-test".into()),
            ..Default::default()
        },
        fake.clone(),
    )
    .unwrap();
    assert!(Arc::ptr_eq(
        &registry.get_provider("anthropic").unwrap(),
        &registry.get_provider("@ai-sdk/anthropic").unwrap()
    ));
    let p = provider(fake)
        .with_name("custom-anthropic")
        .with_anthropic_version("2024-01-01");
    let model = Model::new("custom", Arc::new(p));
    let snapshot = model.to_value().unwrap();
    let restored = Model::from_value(snapshot.clone(), &registry).unwrap();
    assert_eq!(restored.to_value().unwrap(), snapshot);
}
#[test]
fn modelsdev_custom_provider_requires_url_and_uses_its_config_envs() {
    let fake = ScriptedTransport::new(vec![]);
    let metadata = ModelsDevProvider {
        id: "custom".into(),
        api: Some("https://$ACCOUNT.example".into()),
        npm: Some("@ai-sdk/anthropic".into()),
        env: vec!["CUSTOM_API_KEY".into(), "ACCOUNT".into()],
    };
    let p = AnthropicCompatibleProvider::from_modelsdev(
        &metadata,
        ProviderConfig {
            env: [
                ("CUSTOM_API_KEY".into(), "key".into()),
                ("ACCOUNT".into(), "acct".into()),
            ]
            .into(),
            ..Default::default()
        },
        fake.clone(),
    )
    .unwrap();
    assert_eq!(p.name(), "custom");
    assert_eq!(p.base_url(), "https://acct.example");
    assert!(p.is_configured());
    let metadata = ModelsDevProvider {
        api: None,
        ..metadata
    };
    assert!(
        AnthropicCompatibleProvider::from_modelsdev(&metadata, ProviderConfig::default(), fake)
            .is_err()
    );
}

#[tokio::test]
async fn configured_provider_exposes_a_transport_for_owned_protocol_streams() {
    let fake = ScriptedTransport::events(complete(vec![], Some("end_turn")));
    let p = suprnova_ai::providers::ConfiguredProvider::new(
        "proxy",
        ProviderConfig {
            default_base_url: "https://anthropic.test".into(),
            ..Default::default()
        },
        Arc::new(suprnova_ai::providers::anthropic::AnthropicMessagesProtocol),
        Some(fake),
    );
    let m = Model::new("claude", Arc::new(p));
    let mut s = suprnova_ai::stream(&m, vec![], Default::default());
    use futures::StreamExt;
    while let Some(event) = s.next().await {
        event.unwrap();
    }
    assert!(s.ended());
}
#[tokio::test]
async fn malformed_model_lists_are_response_errors() {
    for value in [json!({}), json!({"data":[{}]}), json!({"data":null})] {
        let p = provider(ScriptedTransport::new(vec![Ok(response(200, value))]));
        let suprnova_ai::Error::Provider(err) = p.list_models().await.unwrap_err() else {
            panic!()
        };
        assert_eq!(err.kind, suprnova_ai::errors::ProviderErrorKind::Response);
    }
}
