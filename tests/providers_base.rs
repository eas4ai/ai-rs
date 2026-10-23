use async_trait::async_trait;
use futures::StreamExt;
use std::{collections::BTreeMap, sync::Arc};
use suprnova_ai::providers::{
    ConfiguredProvider, ModelsDevProvider,
    base::{provider_base_url, provider_config},
    operations::*,
};
use suprnova_ai::{
    Error, Model, Provider, ProviderConfig, ProviderProtocol, ProviderRegistry, StreamOptions,
};

struct EmptyProtocol;
#[async_trait]
impl ProviderProtocol for EmptyProtocol {
    fn protocol_class_id(&self) -> &str {
        "empty"
    }
}
struct EmptyProvider {
    config: ProviderConfig,
}
#[async_trait]
impl Provider for EmptyProvider {
    fn name(&self) -> &str {
        "empty"
    }
    fn config(&self) -> &ProviderConfig {
        &self.config
    }
}
fn empty_model() -> Model {
    Model {
        id: "test".into(),
        provider: Arc::new(EmptyProvider {
            config: ProviderConfig::default(),
        }),
        protocol: None,
    }
}
fn unsupported<T>(result: Result<T, Error>) {
    assert!(matches!(result, Err(Error::UnsupportedOperation(_))));
}

#[test]
fn explicit_provider_configuration_matches_base_contract() {
    let mut config = ProviderConfig {
        default_base_url: "https://${ACCOUNT}.example/$REGION".into(),
        api_key_env: Some("API_KEY".into()),
        base_url_env: Some("BASE_URL".into()),
        config_envs: vec!["ACCOUNT".into(), "REGION".into()],
        ..Default::default()
    };
    assert!(!config.is_configured());
    config.env = BTreeMap::from([
        ("API_KEY".into(), "key".into()),
        ("ACCOUNT".into(), "acct".into()),
        ("REGION".into(), "west".into()),
    ]);
    assert!(config.is_configured());
    assert_eq!(config.base_url(), "https://acct.example/west");
    assert_eq!(config.api_key(), Some("key"));
    config.api_key_value = Some("explicit".into());
    config
        .env
        .insert("BASE_URL".into(), "https://override.example".into());
    assert_eq!(config.api_key(), Some("explicit"));
    assert_eq!(config.base_url(), "https://override.example");
    config.env.insert("BASE_URL".into(), String::new());
    assert_eq!(config.base_url(), "https://acct.example/west");
    config.api_key_value = Some(String::new());
    assert_eq!(config.api_key(), Some(""));
    assert!(!config.is_configured());
    let debug = format!("{config:?}");
    assert!(!debug.contains("explicit") && !debug.contains("acct"));
}

#[test]
fn metadata_separates_credentials_and_endpoint_template_values() {
    let metadata = ModelsDevProvider {
        id: "databricks".into(),
        npm: Some("adapter".into()),
        api: Some("https://${DATABRICKS_HOST}/$REGION".into()),
        env: vec!["DATABRICKS_HOST".into(), "DATABRICKS_TOKEN".into()],
    };
    let (key, config) = provider_config(&metadata);
    assert_eq!(key.as_deref(), Some("DATABRICKS_TOKEN"));
    assert_eq!(config, vec!["DATABRICKS_HOST", "REGION"]);
    assert_eq!(provider_base_url(&metadata), metadata.api.as_deref());
    let provider = ConfiguredProvider::from_modelsdev_provider(
        &metadata,
        ProviderConfig::default(),
        Arc::new(EmptyProtocol),
        None,
    );
    assert_eq!(
        provider.config().api_key_env.as_deref(),
        Some("DATABRICKS_TOKEN")
    );
    assert_eq!(provider.base_url(), metadata.api.unwrap());
    assert!(provider.client().is_err());
}

#[test]
fn registry_rejects_duplicate_handles() {
    let model = empty_model();
    let mut registry = ProviderRegistry::new();
    registry.register("custom", model.provider.clone()).unwrap();
    // Registering the same provider again under its handle is allowed.
    registry.register("custom", model.provider.clone()).unwrap();
    assert!(registry.register("custom", empty_model().provider).is_err());
    assert!(registry.register("", model.provider.clone()).is_err());
    assert!(matches!(
        registry.get_provider("missing"),
        Err(Error::UnsupportedProvider(_))
    ));
    let metadata = ModelsDevProvider {
        id: "unknown".into(),
        npm: Some("custom".into()),
        api: None,
        env: vec![],
    };
    assert!(registry.from_modelsdev_provider(&metadata).is_ok());
}

#[tokio::test]
async fn provider_and_protocol_defaults_report_unsupported_operations() {
    let model = empty_model();
    let provider = &model.provider;
    unsupported(
        provider
            .stream(&model, vec![], StreamOptions::default())
            .next()
            .await
            .unwrap(),
    );
    unsupported(
        provider
            .generate(&model, vec![], StreamOptions::default())
            .await,
    );
    unsupported(provider.list_models().await);
    unsupported(provider.probe(&model).await);
    unsupported(provider.embed(&model, vec![], EmbedParams::default()).await);
    unsupported(
        provider
            .generate_image(&model, ImagePrompt::default(), ImageParams::default())
            .await,
    );
    unsupported(
        provider
            .generate_video(&model, VideoPrompt::default(), VideoParams::default())
            .await,
    );
    unsupported(
        provider
            .generate_audio(
                &model,
                AudioPrompt {
                    text: "hi".into(),
                    instructions: None,
                },
                AudioParams::default(),
            )
            .await,
    );
    unsupported(
        provider
            .transcribe(
                &model,
                TranscribeInput::Bytes(vec![]),
                TranscribeParams::default(),
            )
            .await,
    );
    unsupported(
        provider
            .rerank(
                &model,
                Documents::Text(vec![]),
                "q".into(),
                RerankParams::default(),
            )
            .await,
    );
    unsupported(
        provider
            .evaluate(
                &model,
                EvaluationInput::Text("state".into()),
                BTreeMap::new(),
                EvaluationParams::default(),
            )
            .await,
    );
    provider.aclose().await.unwrap();
    let configured = ConfiguredProvider::new(
        "custom",
        ProviderConfig::default(),
        Arc::new(EmptyProtocol),
        None,
    );
    unsupported(
        configured
            .stream(&model, vec![], StreamOptions::default())
            .next()
            .await
            .unwrap(),
    );
    unsupported(
        configured
            .generate(&model, vec![], StreamOptions::default())
            .await,
    );
    unsupported(
        configured
            .embed(&model, vec![], EmbedParams::default())
            .await,
    );
}

#[test]
fn provider_config_accepts_python_null_collections() {
    let config: ProviderConfig =
        serde_json::from_value(serde_json::json!({"headers":null,"env":null,"config_envs":null}))
            .unwrap();
    assert!(config.headers.is_empty());
    assert!(config.env.is_empty());
    assert!(config.config_envs.is_empty());
}

#[test]
fn factories_reject_duplicate_ids_and_missing_or_unknown_discriminators() {
    let mut registry = ProviderRegistry::new();
    registry
        .register_protocol_factory("empty", |_| Ok(Arc::new(EmptyProtocol)))
        .unwrap();
    assert!(
        registry
            .register_protocol_factory("empty", |_| Ok(Arc::new(EmptyProtocol)))
            .is_err()
    );
    registry
        .register_provider_factory("empty", |value, _| {
            Ok(Arc::new(EmptyProvider {
                config: serde_json::from_value(value)?,
            }))
        })
        .unwrap();
    assert!(
        registry
            .register_provider_factory("empty", |_, _| Ok(empty_model().provider))
            .is_err()
    );
    for value in [
        serde_json::json!({}),
        serde_json::json!({"provider_class_id":"missing"}),
    ] {
        assert!(registry.restore_provider(value).is_err());
    }
    for value in [
        serde_json::json!({}),
        serde_json::json!({"protocol_class_id":"missing"}),
    ] {
        assert!(registry.restore_protocol(value).is_err());
    }
    // Without a provider registered under the snapshot's name, nothing supplies its credentials.
    assert!(matches!(
        registry.restore_provider(serde_json::json!({"provider_class_id":"empty","name":"empty"})),
        Err(Error::Configuration(_))
    ));
    registry.register("empty", empty_model().provider).unwrap();
    let provider = registry
        .restore_provider(serde_json::json!({"provider_class_id":"empty","name":"empty"}))
        .unwrap();
    assert_eq!(provider.name(), "empty");
    assert_eq!(
        registry
            .restore_protocol(serde_json::json!({"protocol_class_id":"empty"}))
            .unwrap()
            .protocol_class_id(),
        "empty"
    );
}

#[test]
fn restore_takes_the_endpoint_and_credentials_from_the_registered_provider() {
    let registered = ProviderConfig {
        default_base_url: "https://api.example".into(),
        api_key_value: Some("sk-live".into()),
        api_key_env: Some("EXAMPLE_KEY".into()),
        ..Default::default()
    };
    let mut registry = ProviderRegistry::new();
    registry
        .register(
            "empty",
            Arc::new(EmptyProvider {
                config: registered.clone(),
            }),
        )
        .unwrap();
    for class in ["empty", "other"] {
        registry
            .register_provider_factory(class, |value, _| {
                Ok(Arc::new(EmptyProvider {
                    config: serde_json::from_value(value)?,
                }))
            })
            .unwrap();
    }
    // A changed snapshot, or one saved before the provider moved, restores
    // against the registered endpoint, never its own.
    for snapshot in [
        serde_json::json!({"provider_class_id":"empty","name":"empty","default_base_url":"https://api.example","api_key_env":"EXAMPLE_KEY"}),
        serde_json::json!({
            "provider_class_id": "empty",
            "name": "empty",
            "default_base_url": "https://attacker.example",
            "base_url_env": "OTHER_URL",
            "api_key_env": "OTHER_KEY",
            "config_envs": ["HOST"],
            "api_key_value": "sk-snapshot"
        }),
    ] {
        let restored = registry.restore_provider(snapshot).unwrap();
        assert_eq!(*restored.config(), registered);
        assert_eq!(restored.base_url(), "https://api.example");
        assert_eq!(restored.api_key(), Some("sk-live"));
    }
    match registry.restore_provider(
        serde_json::json!({"provider_class_id":"other","name":"empty","default_base_url":"https://api.example"}),
    ) {
        Err(Error::Configuration(message)) => assert!(message.contains("class"), "{message}"),
        Err(other) => panic!("expected a configuration error, got {other:?}"),
        Ok(_) => panic!("a snapshot of another provider class was restored with the live key"),
    }
}

#[test]
fn operation_payload_defaults_match_python() {
    let warning: Warning = serde_json::from_value(serde_json::json!({})).unwrap();
    assert_eq!(warning.kind, "other");
    assert_eq!(Warning::default().kind, "other");
    let image: ImageParams = serde_json::from_value(serde_json::json!({})).unwrap();
    let video: VideoParams = serde_json::from_value(serde_json::json!({})).unwrap();
    assert_eq!(image.n, 1);
    assert_eq!(video.n, 1);
}

#[tokio::test]
async fn evaluation_rubrics_validate_before_protocol_dispatch() {
    for value in [
        serde_json::json!({"type":"choice","instructions":"choose","criteria":{}}),
        serde_json::json!({"type":"score","instructions":"score","criteria":["one"]}),
    ] {
        assert!(serde_json::from_value::<Question>(value).is_err());
    }
    let valid: Question = serde_json::from_value(
        serde_json::json!({"type":"score","instructions":"score","criteria":["low","high"]}),
    )
    .unwrap();
    valid.validate().unwrap();
    let model = empty_model();
    let questions = BTreeMap::from([(
        "bad".into(),
        Question::Choice {
            instructions: EvaluationInput::Text("choose".into()),
            criteria: BTreeMap::new(),
        },
    )]);
    assert!(matches!(
        model
            .provider
            .evaluate(
                &model,
                EvaluationInput::Text("state".into()),
                questions,
                EvaluationParams::default()
            )
            .await,
        Err(Error::Validation(_))
    ));
    assert!(
        serde_json::to_value(ProviderConfig::default())
            .unwrap()
            .get("api_key_value")
            .is_none()
    );
}
