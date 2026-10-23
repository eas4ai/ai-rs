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

//! Tests for VTX-001 to VTX-015 in docs/spec/vertex.md. Each test name
//! starts with the requirement it checks; scripts/check-vertex.sh selects
//! them by it. A recording transport keeps every request the crate sends and
//! answers it with a reply written in the test, and an access token source
//! or fake google-cloud-auth credentials sign in, so no request leaves the
//! machine and no Google Cloud account is needed. The tests show the crate
//! sends what the AI SDK for TypeScript 7.0.148 sends; they cannot show that
//! Vertex AI still accepts it.

mod google_support;

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use futures::future::BoxFuture;
use google_support::*;
use serde_json::{Value, json};
use suprnova_ai::{
    Error, Model, ProviderConfig, ProviderRegistry, StreamOptions,
    models::params::*,
    ops,
    providers::{
        google::{self, interactions},
        google_vertex::{self, AccessTokenSource},
        operations::*,
    },
    types::{
        Message, MessageRole, Part, Tool,
        messages::{ContentOutput, ContentPart, FileData, FilePart, ToolCallPart, ToolResultPart},
        tools::ToolSpec,
    },
    user_message,
};

const PROJECT: &str = "p1";
const BASE: &str =
    "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/p1/locations/us-central1";
const MIB: usize = 1024 * 1024;

fn env(pairs: &[(&str, &str)]) -> ProviderConfig {
    ProviderConfig {
        env: pairs
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect(),
        ..Default::default()
    }
}

/// A configuration with the project `p1` at `location`.
fn located(location: &str) -> ProviderConfig {
    env(&[
        ("GOOGLE_VERTEX_PROJECT", PROJECT),
        ("GOOGLE_VERTEX_LOCATION", location),
    ])
}

fn express() -> ProviderConfig {
    ProviderConfig {
        api_key_value: Some("vx-key".into()),
        ..Default::default()
    }
}

/// An access token source that counts its calls, or fails.
struct Tokens {
    token: Option<String>,
    calls: AtomicUsize,
}

impl AccessTokenSource for Tokens {
    fn access_token(&self) -> BoxFuture<'_, Result<String, Error>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let token = self.token.clone();
        Box::pin(async move {
            token.ok_or_else(|| Error::Validation("secret-token-from-source".into()))
        })
    }
}

fn tokens(token: &str) -> Arc<Tokens> {
    Arc::new(Tokens {
        token: Some(token.into()),
        calls: AtomicUsize::new(0),
    })
}

fn registry_of(provider: google_vertex::VertexProvider) -> ProviderRegistry {
    let mut registry = ProviderRegistry::new();
    google_vertex::register_provider(&mut registry, provider).unwrap();
    registry
}

fn model_of(id: &str, provider: google_vertex::VertexProvider) -> Model {
    Model::new(
        id,
        registry_of(provider).get_provider("google-vertex").unwrap(),
    )
}

/// A model of a provider at `us-central1` that signs in with the token `t1`.
fn signed(id: &str, transport: Arc<Recorder>) -> Model {
    model_of(
        id,
        google_vertex::provider(located("us-central1"), transport)
            .with_access_token_source(tokens("t1")),
    )
}

fn express_model(id: &str, transport: Arc<Recorder>) -> Model {
    model_of(id, google_vertex::provider(express(), transport))
}

fn hi() -> Vec<Message> {
    vec![user_message("hi")]
}

fn tool(name: &str) -> Tool {
    Tool::function(
        name,
        ToolSpec {
            description: Some(format!("The {name} tool")),
            params: json!({"type":"object","properties":{"q":{"type":"string"}}})
                .try_into()
                .unwrap(),
        },
    )
}

fn options_under(key: &str, value: Value) -> StreamOptions {
    google_options(key, value)
}

fn configuration(error: &Error) {
    assert!(matches!(error, Error::Configuration(_)), "{error:?}");
}

/// An error's text with every source, as a log would show it.
fn chain(error: &Error) -> String {
    let mut text = format!("{error} {error:?}");
    let mut source = std::error::Error::source(error);
    while let Some(next) = source {
        text.push_str(&format!(" {next} {next:?}"));
        source = next.source();
    }
    text
}

// VTX-001

#[tokio::test]
async fn vtx_001_requests_go_to_the_host_of_each_location_kind() {
    for (location, host) in [
        ("global", "aiplatform.googleapis.com"),
        ("eu", "aiplatform.eu.rep.googleapis.com"),
        ("us", "aiplatform.us.rep.googleapis.com"),
        ("us-central1", "us-central1-aiplatform.googleapis.com"),
    ] {
        let transport = Recorder::new([text_reply("hi")]);
        let model = model_of(
            "gemini-2.5-flash",
            google_vertex::provider(located(location), transport.clone())
                .with_access_token_source(tokens("t1")),
        );
        let (handle, events) = call(&model, hi(), StreamOptions::default()).await;
        ok(&events);
        assert_eq!(handle.text(), "hi");
        let request = transport.request();
        assert_eq!(
            request.url,
            format!(
                "https://{host}/v1beta1/projects/p1/locations/{location}/publishers/google/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
            )
        );
        assert_eq!(
            header(&request, "authorization").as_deref(),
            Some("Bearer t1")
        );
        assert_eq!(header(&request, "x-goog-api-key"), None);
    }
    let transport = Recorder::new([text_reply("hi")]);
    let (_, events) = call(
        &signed("endpoints/123", transport.clone()),
        hi(),
        StreamOptions::default(),
    )
    .await;
    ok(&events);
    assert_eq!(
        transport.request().url,
        format!("{BASE}/endpoints/123:streamGenerateContent?alt=sse")
    );
}

#[tokio::test]
async fn vtx_001_an_express_key_goes_to_the_global_publisher_in_a_header() {
    for config in [express(), env(&[("GOOGLE_VERTEX_API_KEY", "vx-key")])] {
        let transport = Recorder::new([text_reply("hi")]);
        let model = model_of(
            "gemini-2.5-flash",
            google_vertex::provider(config, transport.clone()),
        );
        let (_, events) = call(&model, hi(), StreamOptions::default()).await;
        ok(&events);
        let request = transport.request();
        assert_eq!(
            request.url,
            "https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
        );
        assert_eq!(
            header(&request, "x-goog-api-key").as_deref(),
            Some("vx-key")
        );
        assert_eq!(header(&request, "authorization"), None);
    }
}

#[tokio::test]
async fn vtx_001_a_bad_location_or_a_missing_project_sends_nothing() {
    for config in [
        located("us-central1/x"),
        located("-us"),
        env(&[("GOOGLE_VERTEX_LOCATION", "us-central1")]),
        env(&[("GOOGLE_VERTEX_PROJECT", PROJECT)]),
    ] {
        let transport = Recorder::new([text_reply("hi")]);
        let model = model_of(
            "gemini-2.5-flash",
            google_vertex::provider(config, transport.clone())
                .with_access_token_source(tokens("t1")),
        );
        let (_, events) = call(&model, hi(), StreamOptions::default()).await;
        configuration(error(&events));
        assert_eq!(transport.count(), 0);
    }
}

#[tokio::test]
async fn vtx_001_the_provider_has_both_handles_and_restores_from_a_snapshot() {
    let transport = Recorder::new([text_reply("hi")]);
    let registry = registry_of(
        google_vertex::provider(located("us-central1"), transport.clone())
            .with_access_token_source(tokens("t1")),
    );
    for handle in ["google-vertex", "@ai-sdk/google-vertex"] {
        assert_eq!(
            registry.get_provider(handle).unwrap().name(),
            "google-vertex"
        );
    }
    let model = Model::new(
        "gemini-2.5-flash",
        registry.get_provider("google-vertex").unwrap(),
    );
    let snapshot = model.to_value().unwrap();
    let restored = Model::from_value(snapshot, &registry).unwrap();
    let (_, events) = call(&restored, hi(), StreamOptions::default()).await;
    ok(&events);
    let request = transport.request();
    assert_eq!(
        request.url,
        format!("{BASE}/publishers/google/models/gemini-2.5-flash:streamGenerateContent?alt=sse")
    );
    assert_eq!(
        header(&request, "authorization").as_deref(),
        Some("Bearer t1")
    );
}

#[tokio::test]
async fn vtx_001_express_keys_do_not_run_tuned_models_interactions_chirp_or_transcription() {
    let transport = Recorder::new([]);
    let (_, events) = call(
        &express_model("endpoints/123", transport.clone()),
        hi(),
        StreamOptions::default(),
    )
    .await;
    configuration(error(&events));
    let interactions_model = express_model("gemini-3-flash-preview", transport.clone())
        .with_protocol(interactions::protocol());
    let (_, events) = call(&interactions_model, hi(), StreamOptions::default()).await;
    configuration(error(&events));
    let speech = AudioPrompt {
        text: "hello".into(),
        instructions: None,
    };
    let audio = ops::generate_audio(
        &express_model("chirp-3-hd", transport.clone()),
        speech,
        AudioParams::default(),
        Default::default(),
    )
    .await;
    configuration(&audio.unwrap_err());
    for id in ["gemini-3.5-transcribe", "chirp_3"] {
        let result = ops::transcribe(
            &express_model(id, transport.clone()),
            TranscribeInput::File(
                FilePart::from_bytes(vec![1, 2], Some("audio/wav"), None).unwrap(),
            ),
            TranscribeParams::default(),
            Default::default(),
        )
        .await;
        configuration(&result.unwrap_err());
    }
    let stream = ops::stream_transcription(
        &express_model("gemini-3.5-transcribe-live", transport.clone()),
        futures::stream::empty().boxed(),
        AudioFormat::pcm(16000),
        StreamTranscribeParams::default(),
        Default::default(),
    )
    .await;
    configuration(&stream.err().unwrap());
    assert_eq!(transport.count(), 0);
}

use futures::StreamExt;

// VTX-002

#[tokio::test]
async fn vtx_002_a_token_source_signs_every_request_and_configured_headers_win() {
    let transport = Recorder::new([text_reply("a"), text_reply("b")]);
    let source = tokens("t1");
    let model = model_of(
        "gemini-2.5-flash",
        google_vertex::provider(located("us-central1"), transport.clone())
            .with_access_token_source(source.clone()),
    );
    for _ in 0..2 {
        let (_, events) = call(&model, hi(), StreamOptions::default()).await;
        ok(&events);
    }
    assert_eq!(source.calls.load(Ordering::SeqCst), 2, "a token was kept");
    for request in transport.all() {
        assert_eq!(
            header(&request, "authorization").as_deref(),
            Some("Bearer t1")
        );
    }

    let transport = Recorder::new([text_reply("a")]);
    let mut config = located("us-central1");
    config.headers = [("Authorization".to_owned(), "Bearer mine".to_owned())].into();
    let model = model_of(
        "gemini-2.5-flash",
        google_vertex::provider(config, transport.clone()).with_access_token_source(tokens("t1")),
    );
    let (_, events) = call(&model, hi(), StreamOptions::default()).await;
    ok(&events);
    let values: Vec<String> = transport
        .request()
        .headers
        .iter()
        .filter(|(n, _)| n.eq_ignore_ascii_case("authorization"))
        .map(|(_, v)| v.clone())
        .collect();
    assert_eq!(values, vec!["Bearer mine".to_owned()]);
}

#[tokio::test]
async fn vtx_002_a_call_without_credentials_fails_before_a_request() {
    let transport = Recorder::new([text_reply("a")]);
    let model = model_of(
        "gemini-2.5-flash",
        google_vertex::provider(located("us-central1"), transport.clone()),
    );
    let (_, events) = call(&model, hi(), StreamOptions::default()).await;
    let failure = error(&events);
    configuration(failure);
    if cfg!(feature = "google-auth") {
        assert!(failure.to_string().contains("service account"), "{failure}");
    } else {
        assert!(
            failure.to_string().contains("google-auth feature"),
            "{failure}"
        );
    }
    assert_eq!(transport.count(), 0);
}

#[tokio::test]
async fn vtx_002_a_failed_token_source_hides_its_message() {
    let transport = Recorder::new([text_reply("a")]);
    let source = Arc::new(Tokens {
        token: None,
        calls: AtomicUsize::new(0),
    });
    let model = model_of(
        "gemini-2.5-flash",
        google_vertex::provider(located("us-central1"), transport.clone())
            .with_access_token_source(source),
    );
    let options = StreamOptions {
        retry: suprnova_ai::RetryPolicy::none(),
        ..Default::default()
    };
    let (_, events) = call(&model, hi(), options).await;
    let failure = error(&events);
    assert!(matches!(failure, Error::Provider(_)), "{failure:?}");
    assert!(
        !chain(failure).contains("secret-token-from-source"),
        "{}",
        chain(failure)
    );
    assert_eq!(transport.count(), 0);
}

#[tokio::test]
async fn vtx_002_a_snapshot_holds_no_credential() {
    let mut config = located("us-central1");
    config.env.insert(
        "GOOGLE_CLIENT_EMAIL".into(),
        "sa@p1.iam.gserviceaccount.com".into(),
    );
    config.env.insert(
        "GOOGLE_PRIVATE_KEY".into(),
        "-----BEGIN PRIVATE KEY-----\nc2VjcmV0\n-----END PRIVATE KEY-----\n".into(),
    );
    config.api_key_value = Some("vx-key".into());
    let model = model_of(
        "gemini-2.5-flash",
        google_vertex::provider(config, Recorder::new([])).with_access_token_source(tokens("t1")),
    );
    let snapshot = model.to_value().unwrap().to_string();
    for secret in ["c2VjcmV0", "PRIVATE KEY", "vx-key", "t1", "sa@p1"] {
        assert!(!snapshot.contains(secret), "{secret} in {snapshot}");
    }
}

#[cfg(feature = "google-auth")]
mod google_auth {
    use google_cloud_auth::credentials::{
        CacheableResource, Credentials, CredentialsProvider, EntityTag,
    };

    use super::*;

    /// Credentials that give fixed headers and count their calls.
    #[derive(Debug)]
    struct Fake(Arc<AtomicUsize>);

    impl CredentialsProvider for Fake {
        async fn headers(
            &self,
            _extensions: http::Extensions,
        ) -> Result<CacheableResource<http::HeaderMap>, google_cloud_auth::errors::CredentialsError>
        {
            self.0.fetch_add(1, Ordering::SeqCst);
            let mut headers = http::HeaderMap::new();
            headers.insert("authorization", "Bearer t2".parse().unwrap());
            headers.insert("x-goog-user-project", "billing-1".parse().unwrap());
            Ok(CacheableResource::New {
                entity_tag: EntityTag::new(),
                data: headers,
            })
        }
        async fn universe_domain(&self) -> Option<String> {
            None
        }
    }

    #[tokio::test]
    async fn vtx_002_credentials_give_every_header_on_each_request() {
        let calls = Arc::new(AtomicUsize::new(0));
        let transport = Recorder::new([text_reply("a"), text_reply("b")]);
        let model = model_of(
            "gemini-2.5-flash",
            google_vertex::provider(located("us-central1"), transport.clone())
                .with_credentials(Credentials::from(Fake(calls.clone()))),
        );
        for _ in 0..2 {
            let (_, events) = call(&model, hi(), StreamOptions::default()).await;
            ok(&events);
        }
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        for request in transport.all() {
            assert_eq!(
                header(&request, "authorization").as_deref(),
                Some("Bearer t2")
            );
            assert_eq!(
                header(&request, "x-goog-user-project").as_deref(),
                Some("billing-1")
            );
        }
    }

    fn base64url(text: &str) -> Value {
        use base64::Engine as _;
        let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(text)
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    fn service_account(key: &str) -> ProviderConfig {
        let mut config = located("us-central1");
        config.env.extend([
            (
                "GOOGLE_CLIENT_EMAIL".to_owned(),
                "sa@p1.iam.gserviceaccount.com".to_owned(),
            ),
            ("GOOGLE_PRIVATE_KEY".to_owned(), key.to_owned()),
            ("GOOGLE_PRIVATE_KEY_ID".to_owned(), "key-1".to_owned()),
        ]);
        config
    }

    #[tokio::test]
    async fn vtx_002_a_service_account_signs_a_jwt_with_its_key_id_email_and_scope() {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let key = include_str!("fixtures/vertex-test-service-account-key.pem");
        let transport = Recorder::new([text_reply("a")]);
        let model = model_of(
            "gemini-2.5-flash",
            google_vertex::provider(service_account(key), transport.clone()),
        );
        let (_, events) = call(&model, hi(), StreamOptions::default()).await;
        ok(&events);
        let authorization = header(&transport.request(), "authorization").unwrap();
        let token = authorization.strip_prefix("Bearer ").unwrap();
        let parts: Vec<&str> = token.split('.').collect();
        assert_eq!(parts.len(), 3, "not a JWT");
        let jwt_header = base64url(parts[0]);
        assert_eq!(jwt_header["kid"], "key-1");
        assert_eq!(jwt_header["alg"], "RS256");
        let claims = base64url(parts[1]);
        assert_eq!(claims["iss"], "sa@p1.iam.gserviceaccount.com");
        assert_eq!(
            claims["scope"],
            "https://www.googleapis.com/auth/cloud-platform"
        );
    }

    #[tokio::test]
    async fn vtx_002_a_bad_service_account_key_fails_without_quoting_it() {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let key =
            "-----BEGIN PRIVATE KEY-----\nc2VjcmV0LWtleS1ieXRlcw==\n-----END PRIVATE KEY-----\n";
        let transport = Recorder::new([text_reply("a")]);
        let model = model_of(
            "gemini-2.5-flash",
            google_vertex::provider(service_account(key), transport.clone()),
        );
        let options = StreamOptions {
            retry: suprnova_ai::RetryPolicy::none(),
            ..Default::default()
        };
        let (_, events) = call(&model, hi(), options).await;
        let failure = error(&events);
        assert!(matches!(failure, Error::Provider(_)), "{failure:?}");
        assert!(!chain(failure).contains("c2VjcmV0"), "{}", chain(failure));
        assert_eq!(transport.count(), 0);
    }
}

// VTX-003

async fn vertex_body(id: &str, messages: Vec<Message>, options: StreamOptions) -> Value {
    let transport = Recorder::new([text_reply("ok")]);
    let (_, events) = call(&signed(id, transport.clone()), messages, options).await;
    ok(&events);
    transport.body()
}

async fn vertex_rejected(id: &str, options: StreamOptions) {
    let transport = Recorder::new([text_reply("ok")]);
    let (_, events) = call(&signed(id, transport.clone()), hi(), options).await;
    assert!(
        matches!(error(&events), Error::Validation(_)),
        "{:?}",
        error(&events)
    );
    assert_eq!(transport.count(), 0);
}

#[tokio::test]
async fn vtx_003_options_come_from_each_vertex_key() {
    for key in ["googleVertex", "vertex", "google"] {
        let body = vertex_body(
            "gemini-2.5-flash",
            hi(),
            options_under(key, json!({"thinkingConfig": {"thinkingBudget": 64}})),
        )
        .await;
        assert_eq!(
            body["generationConfig"]["thinkingConfig"]["thinkingBudget"], 64,
            "{key}"
        );
    }
}

#[tokio::test]
async fn vtx_003_signatures_are_kept_under_both_vertex_keys() {
    let transport = Recorder::new([Reply::sse(&[json!({
        "candidates": [{"content": {"role": "model", "parts": [{"text": "hi", "thoughtSignature": "sig-1"}]}, "finishReason": "STOP"}]
    })])]);
    let (handle, events) = call(
        &signed("gemini-2.5-flash", transport),
        hi(),
        StreamOptions::default(),
    )
    .await;
    ok(&events);
    let Part::Text(text) = &handle.message().parts[0] else {
        panic!("expected text");
    };
    let metadata = text.provider_metadata.as_ref().unwrap();
    for key in ["googleVertex", "vertex"] {
        assert_eq!(metadata[key]["thoughtSignature"], "sig-1", "{key}");
    }
}

#[tokio::test]
async fn vtx_003_tool_calls_and_results_carry_no_id() {
    let history = vec![
        user_message("hi"),
        Message::new(
            MessageRole::Assistant,
            vec![Part::ToolCall(ToolCallPart::new(
                "c1",
                "lookup",
                r#"{"q":"x"}"#,
            ))],
        ),
        Message::new(
            MessageRole::Tool,
            vec![Part::ToolResult(ToolResultPart::new(
                "c1",
                "lookup",
                json!({"v": 1}),
            ))],
        ),
    ];
    let body = vertex_body("gemini-2.5-flash", history, StreamOptions::default()).await;
    let call_part = &body["contents"][1]["parts"][0]["functionCall"];
    assert_eq!(call_part["name"], "lookup");
    assert!(call_part.get("id").is_none(), "{call_part}");
    let response = &body["contents"][2]["parts"][0]["functionResponse"];
    assert_eq!(response["name"], "lookup");
    assert!(response.get("id").is_none(), "{response}");
}

fn penalties() -> StreamOptions {
    let sampling = SamplerParams {
        repetition_penalty: Some(RepetitionPenaltyParams {
            frequency_penalty: ProviderValue::Value(Some(0.1)),
            presence_penalty: ProviderValue::Value(Some(0.2)),
            ..Default::default()
        }),
        ..Default::default()
    };
    params(|p| p.sampling = ProviderValue::Value(sampling))
}

#[tokio::test]
async fn vtx_003_gemini_2_5_takes_penalties_and_service_tiers_fail() {
    let body = vertex_body("gemini-2.5-flash", hi(), penalties()).await;
    assert_eq!(body["generationConfig"]["frequencyPenalty"], 0.1);
    assert_eq!(body["generationConfig"]["presencePenalty"], 0.2);
    vertex_rejected(
        "gemini-2.5-flash",
        options_under("googleVertex", json!({"serviceTier": "flex"})),
    )
    .await;
    vertex_rejected(
        "gemini-2.5-flash",
        params(|p| {
            p.provider_service = Some(ProviderServiceParams {
                service_tier: ProviderValue::Value("flex".into()),
            })
        }),
    )
    .await;
}

#[tokio::test]
async fn vtx_003_request_types_go_out_as_headers_and_image_options_are_sent() {
    let transport = Recorder::new([text_reply("ok")]);
    let options = options_under(
        "googleVertex",
        json!({"sharedRequestType": "priority", "requestType": "shared"}),
    );
    let (_, events) = call(
        &signed("gemini-2.5-flash", transport.clone()),
        hi(),
        options,
    )
    .await;
    ok(&events);
    let request = transport.request();
    assert_eq!(
        header(&request, "x-vertex-ai-llm-shared-request-type").as_deref(),
        Some("priority")
    );
    assert_eq!(
        header(&request, "x-vertex-ai-llm-request-type").as_deref(),
        Some("shared")
    );

    let image = json!({
        "personGeneration": "ALLOW_ADULT",
        "prominentPeople": "BLOCK_PROMINENT_PEOPLE",
        "imageOutputOptions": {"mimeType": "image/jpeg", "compressionQuality": 80},
    });
    let body = vertex_body(
        "gemini-2.5-flash-image",
        hi(),
        options_under("googleVertex", json!({"imageConfig": image})),
    )
    .await;
    assert_eq!(body["generationConfig"]["imageConfig"], image);
}

#[tokio::test]
async fn vtx_003_tools_and_file_urls_take_their_vertex_shapes() {
    let options = StreamOptions {
        tools: vec![
            tool("lookup"),
            google::tools::google_search(json!({})).unwrap(),
        ],
        ..Default::default()
    };
    let body = vertex_body("gemini-3-flash-preview", hi(), options).await;
    assert!(
        body.pointer("/toolConfig/includeServerSideToolInvocations")
            .is_none(),
        "{body}"
    );

    let options = StreamOptions {
        tools: vec![
            google::tools::enterprise_web_search(),
            google::tools::vertex_rag_store(
                json!({"ragCorpus": "projects/p1/locations/us-central1/ragCorpora/c1", "topK": 3}),
            )
            .unwrap(),
        ],
        ..Default::default()
    };
    let body = vertex_body("gemini-2.5-flash", hi(), options).await;
    assert_eq!(
        body["tools"],
        json!([
            {"enterpriseWebSearch": {}},
            {"retrieval": {"vertex_rag_store": {"rag_resources": {"rag_corpus": "projects/p1/locations/us-central1/ragCorpora/c1"}, "similarity_top_k": 3}}},
        ])
    );

    let files = vec![Message::new(
        MessageRole::User,
        vec![
            Part::File(
                FilePart::from_url("https://example.com/a.pdf", Some("application/pdf")).unwrap(),
            ),
            Part::File(FilePart::from_url("gs://bucket/b.png", Some("image/png")).unwrap()),
        ],
    )];
    let body = vertex_body("gemini-2.5-flash", files, StreamOptions::default()).await;
    assert_eq!(
        body["contents"][0]["parts"],
        json!([
            {"fileData": {"fileUri": "https://example.com/a.pdf", "mimeType": "application/pdf"}},
            {"fileData": {"fileUri": "gs://bucket/b.png", "mimeType": "image/png"}},
        ])
    );
}

#[tokio::test]
async fn vtx_003_large_vertex_responses_are_read_whole() {
    let data = "A".repeat(20 * MIB);
    let image_reply = || {
        Reply::json(
            json!({"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": data}}]}, "finishReason": "STOP"}]}),
        )
    };
    let transport = Recorder::with_chunks([image_reply()], MIB);
    let message = suprnova_ai::generate(
        &signed("gemini-2.5-flash", transport),
        hi(),
        StreamOptions::default(),
    )
    .await
    .unwrap();
    assert!(
        message.parts.iter().any(
            |p| matches!(p, Part::File(f) if f.data.as_text().map(str::len) == Some(20 * MIB))
        )
    );

    let transport = Recorder::with_chunks([image_reply()], MIB);
    let prompt = ImagePrompt {
        text: Some("a cat".into()),
        images: vec![],
        mask: None,
    };
    let item = ops::generate_image(
        &signed("gemini-2.5-flash-image", transport),
        prompt,
        ImageParams::default(),
        Default::default(),
    )
    .await
    .unwrap();
    assert_eq!(item.value.len(), 1);

    let speech = Reply::json(
        json!({"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "audio/L16;rate=24000", "data": data}}]}}]}),
    );
    let transport = Recorder::with_chunks([speech], MIB);
    let prompt = AudioPrompt {
        text: "hello".into(),
        instructions: None,
    };
    let item = ops::generate_audio(
        &signed("gemini-2.5-flash-tts", transport),
        prompt,
        AudioParams::default(),
        Default::default(),
    )
    .await
    .unwrap();
    assert!(item.value[0].data.as_bytes().unwrap().len() > 15 * MIB);

    let video = Reply::json(
        json!({"done": true, "response": {"videos": [{"bytesBase64Encoded": data, "mimeType": "video/mp4"}]}}),
    );
    let transport = Recorder::with_chunks([video], MIB);
    let job = VideoJob {
        provider: "google-vertex".into(),
        model_id: "veo-3.1-generate-preview".into(),
        operation: json!({"operationName": "op1"}),
    };
    let status = ops::video_status(
        &signed("veo-3.1-generate-preview", transport),
        &job,
        Default::default(),
    )
    .await
    .unwrap();
    assert!(matches!(status, VideoStatus::Completed(_)));
}

// VTX-004

#[tokio::test]
async fn vtx_004_streamed_arguments_join_to_one_tool_call() {
    let chunk = |call: Value| json!({"candidates": [{"content": {"role": "model", "parts": [{"functionCall": call}]}}]});
    let transport = Recorder::new([Reply::sse(&[
        chunk(json!({"name": "get_weather", "willContinue": true})),
        chunk(
            json!({"partialArgs": [{"jsonPath": "$.location", "stringValue": "Bos", "willContinue": true}], "willContinue": true}),
        ),
        chunk(
            json!({"partialArgs": [{"jsonPath": "$.location", "stringValue": "ton"}], "willContinue": true}),
        ),
        chunk(
            json!({"partialArgs": [{"jsonPath": "$.units", "stringValue": "metric"}], "willContinue": true}),
        ),
        chunk(
            json!({"partialArgs": [{"jsonPath": "$.recipe.ingredients[0].name", "stringValue": "Noodles"}], "willContinue": true}),
        ),
        chunk(json!({})),
        json!({"candidates": [{"content": {"role": "model", "parts": []}, "finishReason": "STOP"}]}),
    ])]);
    let options = StreamOptions {
        tools: vec![tool("get_weather")],
        ..options_under("googleVertex", json!({"streamFunctionCallArguments": true}))
    };
    let (handle, events) = call(
        &signed("gemini-3-flash-preview", transport.clone()),
        hi(),
        options,
    )
    .await;
    ok(&events);
    assert_eq!(
        transport.body()["toolConfig"]["functionCallingConfig"]["streamFunctionCallArguments"],
        true
    );
    let deltas: String = events
        .iter()
        .filter_map(|e| match e {
            Ok(suprnova_ai::types::Event::ToolDelta(d)) => Some(d.chunk.clone()),
            _ => None,
        })
        .collect();
    let expected = json!({"location": "Boston", "units": "metric", "recipe": {"ingredients": [{"name": "Noodles"}]}});
    assert_eq!(serde_json::from_str::<Value>(&deltas).unwrap(), expected);
    let calls = handle.message().tool_calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(
        serde_json::from_str::<Value>(&calls[0].tool_args).unwrap(),
        expected
    );
}

// VTX-005

fn tool_result_history(url: &str, media_type: &str) -> Vec<Message> {
    let file = FilePart::from_url(url, Some(media_type)).unwrap();
    vec![
        user_message("hi"),
        Message::new(
            MessageRole::Assistant,
            vec![Part::ToolCall(ToolCallPart::new("c1", "shot", "{}"))],
        ),
        Message::new(
            MessageRole::Tool,
            vec![Part::ToolResult(ToolResultPart::new(
                "c1",
                "shot",
                ContentOutput::new(vec![ContentPart::File(file)]),
            ))],
        ),
    ]
}

fn with_downloads(id: &str, transport: Arc<Recorder>, downloads: Arc<Recorder>) -> Model {
    model_of(
        id,
        google_vertex::provider(located("us-central1"), transport)
            .with_access_token_source(tokens("t1"))
            .with_download_transport(Arc::new(Downloads(downloads))),
    )
}

#[tokio::test]
async fn vtx_005_tool_result_urls_are_downloaded_except_gcs_files_on_gemini_3() {
    let png = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    let downloads = Recorder::new([Reply::bytes(png).header("content-type", "image/png")]);
    let transport = Recorder::new([text_reply("ok")]);
    let model = with_downloads("gemini-2.5-flash", transport.clone(), downloads.clone());
    let (_, events) = call(
        &model,
        tool_result_history("https://example.com/shot.png", "image/png"),
        StreamOptions::default(),
    )
    .await;
    ok(&events);
    assert_eq!(downloads.request().url, "https://example.com/shot.png");
    let body = transport.body().to_string();
    assert!(body.contains("iVBORw0KGgo="), "{body}");
    assert!(!body.contains("https://example.com/shot.png"), "{body}");

    let downloads = Recorder::new([]);
    let transport = Recorder::new([text_reply("ok")]);
    let model = with_downloads(
        "gemini-3-flash-preview",
        transport.clone(),
        downloads.clone(),
    );
    let (_, events) = call(
        &model,
        tool_result_history("gs://bucket/shot.png", "image/png"),
        StreamOptions::default(),
    )
    .await;
    ok(&events);
    assert_eq!(downloads.count(), 0);
    let body = transport.body().to_string();
    assert!(
        body.contains("\"fileUri\":\"gs://bucket/shot.png\""),
        "{body}"
    );
}

#[tokio::test]
async fn vtx_005_a_download_over_the_limit_sends_no_model_request() {
    let downloads = Recorder::with_chunks(
        [Reply::bytes(vec![0; 8 * MIB]).header("content-type", "image/png")],
        MIB,
    );
    let transport = Recorder::new([text_reply("ok")]);
    let model = with_downloads("gemini-2.5-flash", transport.clone(), downloads.clone());
    let (_, events) = call(
        &model,
        tool_result_history("https://example.com/big.png", "image/png"),
        StreamOptions::default(),
    )
    .await;
    assert!(events.iter().any(Result::is_err));
    assert_eq!(downloads.count(), 1);
    assert_eq!(transport.count(), 0);
}

// VTX-006

fn embed_options(value: Value) -> EmbedParams {
    EmbedParams {
        provider_options: [("googleVertex".to_owned(), value)].into(),
    }
}

#[tokio::test]
async fn vtx_006_gemini_embedding_2_uses_embed_content_one_value_per_call() {
    let reply = || {
        Reply::json(
            json!({"embedding": {"values": [0.1, 0.2]}, "usageMetadata": {"promptTokenCount": 3}}),
        )
    };
    let transport = Recorder::new([reply(), reply()]);
    let options = embed_options(
        json!({"outputDimensionality": 256, "taskType": "RETRIEVAL_DOCUMENT", "title": "Doc", "autoTruncate": false}),
    );
    let item = ops::embed(
        &signed("gemini-embedding-2", transport.clone()),
        vec!["hello".into(), "world".into()],
        options,
        Default::default(),
    )
    .await
    .unwrap();
    let requests = transport.all();
    assert_eq!(requests.len(), 2);
    assert_eq!(
        requests[0].url,
        format!("{BASE}/publishers/google/models/gemini-embedding-2:embedContent")
    );
    assert_eq!(
        transport.nth_body(0),
        json!({"content": {"parts": [{"text": "hello"}]}, "embedContentConfig": {"outputDimensionality": 256, "taskType": "RETRIEVAL_DOCUMENT", "title": "Doc", "autoTruncate": false}})
    );
    assert_eq!(item.value, vec![vec![0.1, 0.2], vec![0.1, 0.2]]);
    assert_eq!(item.usage.unwrap().input_tokens, 6);
}

#[tokio::test]
async fn vtx_006_other_models_use_predict_up_to_250_values() {
    let predictions = |n: usize| {
        let entries: Vec<Value> = (0..n)
            .map(
                |i| json!({"embeddings": {"values": [i as f64], "statistics": {"token_count": 2}}}),
            )
            .collect();
        Reply::json(json!({"predictions": entries}))
    };
    let transport = Recorder::new([predictions(2)]);
    let item = ops::embed(
        &signed("text-embedding-005", transport.clone()),
        vec!["a".into(), "b".into()],
        embed_options(json!({"taskType": "RETRIEVAL_QUERY", "autoTruncate": true, "outputDimensionality": 64})),
        Default::default(),
    )
    .await
    .unwrap();
    assert_eq!(
        transport.request().url,
        format!("{BASE}/publishers/google/models/text-embedding-005:predict")
    );
    assert_eq!(
        transport.body(),
        json!({
            "instances": [{"content": "a", "task_type": "RETRIEVAL_QUERY"}, {"content": "b", "task_type": "RETRIEVAL_QUERY"}],
            "parameters": {"outputDimensionality": 64, "autoTruncate": true},
        })
    );
    assert_eq!(item.value, vec![vec![0.0], vec![1.0]]);
    assert_eq!(item.usage.unwrap().input_tokens, 4);

    let transport = Recorder::new([predictions(250), predictions(1)]);
    let values: Vec<String> = (0..251).map(|i| i.to_string()).collect();
    let item = ops::embed(
        &signed("text-embedding-005", transport.clone()),
        values,
        EmbedParams::default(),
        Default::default(),
    )
    .await
    .unwrap();
    let sizes: Vec<usize> = (0..2)
        .map(|i| transport.nth_body(i)["instances"].as_array().unwrap().len())
        .collect();
    assert_eq!(sizes, vec![250, 1]);
    assert_eq!(item.value.len(), 251);
}

// VTX-007

#[tokio::test]
async fn vtx_007_vertex_images_use_generate_content_with_vertex_metadata() {
    let transport = Recorder::new([Reply::json(json!({
        "candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": "iVBORw=="}}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 1290},
    }))]);
    let prompt = ImagePrompt {
        text: Some("a cat".into()),
        images: vec![],
        mask: None,
    };
    let params = ImageParams {
        aspect_ratio: Some("16:9".into()),
        provider_options: [(
            "googleVertex".to_owned(),
            json!({"imageConfig": {"imageSize": "2K"}}),
        )]
        .into(),
        ..Default::default()
    };
    let item = ops::generate_image(
        &signed("gemini-2.5-flash-image", transport.clone()),
        prompt.clone(),
        params,
        Default::default(),
    )
    .await
    .unwrap();
    assert_eq!(
        transport.request().url,
        format!("{BASE}/publishers/google/models/gemini-2.5-flash-image:generateContent")
    );
    let body = transport.body();
    assert_eq!(body["contents"][0]["parts"][0], json!({"text": "a cat"}));
    assert_eq!(
        body["generationConfig"]["responseModalities"],
        json!(["IMAGE"])
    );
    assert_eq!(
        body["generationConfig"]["imageConfig"],
        json!({"imageSize": "2K", "aspectRatio": "16:9"})
    );
    let metadata = item.provider_metadata.unwrap();
    for key in ["googleVertex", "vertex"] {
        assert_eq!(metadata[key], json!({"images": [{}]}), "{key}");
    }

    let transport = Recorder::new([]);
    let params = ImageParams {
        provider_options: [("googleVertex".to_owned(), json!({"googleSearch": {}}))].into(),
        ..Default::default()
    };
    let error = ops::generate_image(
        &signed("gemini-2.5-flash-image", transport.clone()),
        prompt,
        params,
        Default::default(),
    )
    .await
    .unwrap_err();
    assert!(matches!(error, Error::Validation(_)), "{error:?}");
    assert_eq!(transport.count(), 0);
}

// VTX-008

#[tokio::test]
async fn vtx_008_a_vertex_job_starts_with_its_options_and_reads_its_videos() {
    let name = "projects/p1/locations/us-central1/publishers/google/models/veo-3.1-generate-preview/operations/op1";
    let transport = Recorder::new([
        Reply::json(json!({"name": name})),
        Reply::json(json!({"name": name, "done": true, "response": {"videos": [
            {"bytesBase64Encoded": "AAEC", "mimeType": "video/mp4"},
            {"gcsUri": "gs://b/out/v.mp4"},
        ]}})),
    ]);
    let model = signed("veo-3.1-generate-preview", transport.clone());
    let reference = json!({"image": {"gcsUri": "gs://b/r.png", "mimeType": "image/png"}, "referenceType": "asset"});
    let params = VideoParams {
        generate_audio: Some(true),
        provider_options: [(
            "googleVertex".to_owned(),
            json!({"referenceImages": [reference], "gcsOutputDirectory": "gs://b/out"}),
        )]
        .into(),
        ..Default::default()
    };
    let prompt = VideoPrompt {
        text: Some("waves".into()),
        ..Default::default()
    };
    let started = ops::start_video(&model, prompt, params, Default::default())
        .await
        .unwrap();
    assert_eq!(
        transport.nth(0).url,
        format!("{BASE}/publishers/google/models/veo-3.1-generate-preview:predictLongRunning")
    );
    let body = transport.nth_body(0);
    assert_eq!(body["instances"][0]["referenceImages"], json!([reference]));
    assert_eq!(body["parameters"]["generateAudio"], true);
    assert_eq!(body["parameters"]["gcsOutputDirectory"], "gs://b/out");
    let VideoStatus::Completed(item) =
        ops::video_status(&model, &started.value, Default::default())
            .await
            .unwrap()
    else {
        panic!("not completed");
    };
    let status = transport.nth(1);
    assert_eq!(
        (status.method.as_str(), status.url.as_str()),
        (
            "POST",
            format!(
                "{BASE}/publishers/google/models/veo-3.1-generate-preview:fetchPredictOperation"
            )
            .as_str()
        )
    );
    assert_eq!(transport.nth_body(1), json!({"operationName": name}));
    assert_eq!(item.value[0].data, FileData::Bytes(vec![0, 1, 2]));
    assert_eq!(
        item.value[1].data,
        FileData::Text("gs://b/out/v.mp4".into())
    );
    assert_eq!(item.value[1].media_type, "video/mp4");
    let metadata = item.provider_metadata.unwrap();
    for key in ["googleVertex", "google-vertex", "vertex"] {
        assert_eq!(
            metadata[key],
            json!({"videos": [{"mimeType": "video/mp4"}, {"mimeType": "video/mp4", "gcsUri": "gs://b/out/v.mp4"}]}),
            "{key}"
        );
    }
}

// VTX-009

#[tokio::test]
async fn vtx_009_chirp_voices_go_to_cloud_text_to_speech() {
    for (voice, language, name, code) in [
        (None, None, "en-US-Chirp3-HD-Kore", "en-US"),
        (Some("Puck"), Some("de-DE"), "de-DE-Chirp3-HD-Puck", "de-DE"),
        (
            Some("fr-FR-Chirp3-HD-Aoede"),
            None,
            "fr-FR-Chirp3-HD-Aoede",
            "fr-FR",
        ),
    ] {
        let transport = Recorder::new([Reply::json(json!({"audioContent": "UklGRg=="}))]);
        let prompt = AudioPrompt {
            text: "hello".into(),
            instructions: Some("calm".into()),
        };
        let params = AudioParams {
            voice: voice.map(str::to_owned),
            language: language.map(str::to_owned),
            speed: Some(1.25),
            output_format: Some("mp3".into()),
            ..Default::default()
        };
        let item = ops::generate_audio(
            &signed("chirp-3-hd", transport.clone()),
            prompt,
            params,
            Default::default(),
        )
        .await
        .unwrap();
        let request = transport.request();
        assert_eq!(
            request.url,
            "https://texttospeech.googleapis.com/v1/text:synthesize"
        );
        assert_eq!(
            header(&request, "authorization").as_deref(),
            Some("Bearer t1")
        );
        assert_eq!(
            transport.body(),
            json!({
                "input": {"text": "hello"},
                "voice": {"languageCode": code, "name": name},
                "audioConfig": {"audioEncoding": "LINEAR16", "speakingRate": 1.25},
            })
        );
        assert_eq!(item.value[0].data, FileData::Bytes(b"RIFF".to_vec()));
        assert_eq!(item.value[0].media_type, "audio/wav");
        let features: Vec<&str> = item
            .warnings
            .iter()
            .filter_map(|w| w.feature.as_deref())
            .collect();
        assert_eq!(features, vec!["instructions", "outputFormat"]);
    }
}

#[tokio::test]
async fn vtx_009_gemini_speech_on_vertex_reads_its_options_from_the_vertex_keys() {
    let reply = || {
        Reply::json(
            json!({"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "audio/L16;rate=24000", "data": "AQA="}}]}}]}),
        )
    };
    let prompt = || AudioPrompt {
        text: "hello".into(),
        instructions: None,
    };
    let speakers = json!({"speakerVoiceConfigs": [
        {"speaker": "A", "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}},
        {"speaker": "B", "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Puck"}}},
    ]});
    for key in ["googleVertex", "vertex", "google"] {
        let transport = Recorder::new([reply()]);
        let params = AudioParams {
            provider_options: [(
                key.to_owned(),
                json!({"multiSpeakerVoiceConfig": speakers.clone()}),
            )]
            .into(),
            ..Default::default()
        };
        ops::generate_audio(
            &signed("gemini-2.5-flash-tts", transport.clone()),
            prompt(),
            params,
            Default::default(),
        )
        .await
        .unwrap();
        assert_eq!(
            transport.request().url,
            format!("{BASE}/publishers/google/models/gemini-2.5-flash-tts:generateContent")
        );
        assert_eq!(
            transport.body()["generationConfig"]["speechConfig"]["multiSpeakerVoiceConfig"],
            speakers,
            "{key}"
        );
    }
    let transport = Recorder::new([reply()]);
    let params = AudioParams {
        provider_options: [("googleVertex".to_owned(), json!({"unknownOption": true}))].into(),
        ..Default::default()
    };
    let error = ops::generate_audio(
        &signed("gemini-2.5-flash-tts", transport.clone()),
        prompt(),
        params,
        Default::default(),
    )
    .await
    .unwrap_err();
    assert!(matches!(error, Error::Validation(_)), "{error:?}");
    assert_eq!(transport.count(), 0);
}

// VTX-010

fn audio() -> TranscribeInput {
    TranscribeInput::File(FilePart::from_bytes(vec![1, 2, 3], Some("audio/wav"), None).unwrap())
}

fn transcribe_options(value: Value) -> TranscribeParams {
    TranscribeParams {
        provider_options: [("googleVertex".to_owned(), value)].into(),
    }
}

#[tokio::test]
async fn vtx_010_other_models_go_to_cloud_speech_to_text() {
    let transport = Recorder::new([Reply::json(json!({
        "results": [
            {"alternatives": [{"transcript": "hello there", "words": [
                {"word": "hello", "startOffset": "0.100s", "endOffset": "0.500s"},
                {"word": "there", "startOffset": "0.6s", "endOffset": "1s"},
            ]}], "languageCode": "en-US"},
            {"alternatives": [{"transcript": "general kenobi"}]},
        ],
        "metadata": {"totalBilledDuration": "12.5s"},
    }))]);
    let item = ops::transcribe(
        &signed("chirp_3", transport.clone()),
        audio(),
        transcribe_options(
            json!({"languageCodes": ["en-US"], "enableAutomaticPunctuation": false}),
        ),
        Default::default(),
    )
    .await
    .unwrap();
    let request = transport.request();
    assert_eq!(
        request.url,
        "https://us-central1-speech.googleapis.com/v2/projects/p1/locations/us-central1/recognizers/_:recognize"
    );
    assert_eq!(
        header(&request, "authorization").as_deref(),
        Some("Bearer t1")
    );
    assert_eq!(
        transport.body(),
        json!({
            "config": {"model": "chirp_3", "languageCodes": ["en-US"], "autoDecodingConfig": {}, "features": {"enableWordTimeOffsets": true, "enableAutomaticPunctuation": false}},
            "content": "AQID",
        })
    );
    assert_eq!(item.value.text, "hello there general kenobi");
    assert_eq!(
        item.value.segments,
        vec![
            TranscriptionSegment {
                text: "hello".into(),
                start_second: 0.1,
                end_second: 0.5
            },
            TranscriptionSegment {
                text: "there".into(),
                start_second: 0.6,
                end_second: 1.0
            },
        ]
    );
    assert_eq!(item.value.language.as_deref(), Some("en"));
    assert_eq!(item.value.duration_seconds, Some(12.5));
}

#[tokio::test]
async fn vtx_010_the_region_option_picks_the_host_and_defaults_apply() {
    let transport = Recorder::new([Reply::json(json!({}))]);
    ops::transcribe(
        &signed("chirp_3", transport.clone()),
        audio(),
        transcribe_options(json!({"region": "global"})),
        Default::default(),
    )
    .await
    .unwrap();
    assert_eq!(
        transport.request().url,
        "https://speech.googleapis.com/v2/projects/p1/locations/global/recognizers/_:recognize"
    );
    assert_eq!(
        transport.body()["config"],
        json!({"model": "chirp_3", "languageCodes": ["auto"], "autoDecodingConfig": {}, "features": {"enableWordTimeOffsets": true, "enableAutomaticPunctuation": true}})
    );
    let transport = Recorder::new([]);
    let error = ops::transcribe(
        &signed("chirp_3", transport.clone()),
        audio(),
        transcribe_options(json!({"region": "us/x"})),
        Default::default(),
    )
    .await
    .unwrap_err();
    assert!(matches!(error, Error::Validation(_)), "{error:?}");
    assert_eq!(transport.count(), 0);
}

// VTX-011

#[tokio::test]
async fn vtx_011_a_whole_file_uses_generate_content_on_vertex() {
    let transport = Recorder::new([Reply::json(json!({
        "candidates": [{"content": {"parts": [{"audioTranscription": {"text": "hello", "languageCode": "en-US", "words": [
            {"word": "hello", "startOffset": "1.5s", "endOffset": "2s"},
        ]}}]}}],
        "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 2},
    }))]);
    let item = ops::transcribe(
        &signed("gemini-3.5-transcribe", transport.clone()),
        audio(),
        transcribe_options(json!({"languageCodes": ["en-US"], "wordTimestamp": true})),
        Default::default(),
    )
    .await
    .unwrap();
    assert_eq!(
        transport.request().url,
        format!("{BASE}/publishers/google/models/gemini-3.5-transcribe:generateContent")
    );
    assert_eq!(
        transport.body(),
        json!({
            "contents": [{"role": "user", "parts": [{"inlineData": {"mimeType": "audio/wav", "data": "AQID"}}]}],
            "generationConfig": {"audioTranscriptionConfig": {"languageCodes": ["en-US"], "wordTimestamp": true}},
        })
    );
    assert_eq!(item.value.text, "hello");
    assert_eq!(item.value.language.as_deref(), Some("en-US"));
    assert_eq!(
        item.value.segments,
        vec![TranscriptionSegment {
            text: "hello".into(),
            start_second: 1.5,
            end_second: 2.0
        }]
    );
    assert_eq!(item.usage.unwrap().input_tokens, 4);

    let transport = Recorder::new([Reply::json(json!({
        "candidates": [{"content": {"parts": [{"text": "plain"}, {"audioTranscription": {"text": "spoken"}}]}}],
    }))]);
    let item = ops::transcribe(
        &signed("gemini-3.5-transcribe", transport),
        audio(),
        TranscribeParams::default(),
        Default::default(),
    )
    .await
    .unwrap();
    assert_eq!(item.value.text, "plain");
}

#[tokio::test]
async fn vtx_011_live_transcription_opens_the_vertex_socket_with_the_auth_headers() {
    let socket = ScriptedSocket::new(vec![
        (1, text_message(json!({"setupComplete": {}}))),
        (
            2,
            text_message(json!({"serverContent": {"inputTranscription": {"text": "hi"}}})),
        ),
        (usize::MAX, text_message(json!({}))),
    ]);
    let model = model_of(
        "gemini-3.5-transcribe-live",
        google_vertex::provider(located("us-central1"), Recorder::new([]))
            .with_access_token_source(tokens("t1"))
            .with_websocket_transport(socket.clone()),
    );
    let audio = futures::stream::iter((0..3).map(|_| vec![0u8, 0])).boxed();
    let mut parts = ops::stream_transcription(
        &model,
        audio,
        AudioFormat::pcm(16000),
        StreamTranscribeParams::default(),
        Default::default(),
    )
    .await
    .unwrap();
    while !matches!(
        parts.next().await,
        Some(Ok(TranscriptionPart::Delta { .. }))
    ) {}
    let request = socket.request();
    assert_eq!(
        request.url,
        "wss://us-central1-aiplatform.googleapis.com/ws/google.cloud.aiplatform.v1.LlmBidiService/BidiGenerateContent"
    );
    assert!(
        request
            .headers
            .iter()
            .any(|(n, v)| n.eq_ignore_ascii_case("authorization") && v == "Bearer t1")
    );
    assert!(
        !request
            .headers
            .iter()
            .any(|(n, _)| n.eq_ignore_ascii_case("x-goog-api-key"))
    );
    assert_eq!(
        socket.sent_json()[0]["setup"]["model"],
        "projects/p1/locations/us-central1/publishers/google/models/gemini-3.5-transcribe-live"
    );
}

// VTX-012

#[tokio::test]
async fn vtx_012_interactions_go_to_the_location_with_the_auth_headers() {
    let transport = Recorder::new([Reply::json(json!({
        "id": "int-1", "status": "completed",
        "steps": [{"type": "model_output", "content": [{"type": "text", "text": "ok"}]}],
    }))]);
    let model =
        signed("gemini-3-flash-preview", transport.clone()).with_protocol(interactions::protocol());
    let message = suprnova_ai::generate(&model, hi(), StreamOptions::default())
        .await
        .unwrap();
    assert_eq!(message.text(), "ok");
    let request = transport.request();
    assert_eq!(request.url, format!("{BASE}/interactions"));
    assert_eq!(
        header(&request, "authorization").as_deref(),
        Some("Bearer t1")
    );
    assert_eq!(
        header(&request, "api-revision").as_deref(),
        Some("2026-05-20")
    );
}

// VTX-013

const CLAUDE: &str = "claude-sonnet-4-5@20250929";
const CLAUDE_URL: &str = "https://us-east5-aiplatform.googleapis.com/v1/projects/p1/locations/us-east5/publishers/anthropic/models/claude-sonnet-4-5@20250929";

fn messages_reply(events: &[Value]) -> Reply {
    Reply::sse(events)
}

fn messages_text(text: &str) -> Reply {
    messages_reply(&[
        json!({"type": "message_start", "message": {"id": "msg_1", "model": CLAUDE, "content": [], "stop_reason": null, "usage": {"input_tokens": 3, "output_tokens": 0}}}),
        json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}),
        json!({"type": "content_block_stop", "index": 0}),
        json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}}),
        json!({"type": "message_stop"}),
    ])
}

fn claude(
    config: ProviderConfig,
    transport: Arc<Recorder>,
    downloads: Option<Arc<Recorder>>,
) -> Model {
    let mut provider = google_vertex::anthropic::provider(config, transport)
        .with_access_token_source(tokens("t1"));
    if let Some(downloads) = downloads {
        provider = provider.with_download_transport(Arc::new(Downloads(downloads)));
    }
    let mut registry = ProviderRegistry::new();
    google_vertex::anthropic::register_provider(&mut registry, provider).unwrap();
    for handle in ["google-vertex-anthropic", "@ai-sdk/google-vertex/anthropic"] {
        assert_eq!(
            registry.get_provider(handle).unwrap().name(),
            "google-vertex-anthropic"
        );
    }
    Model::new(
        CLAUDE,
        registry.get_provider("google-vertex-anthropic").unwrap(),
    )
}

fn labelled() -> StreamOptions {
    options_under("vertex", json!({"labels": {"team": "ai"}}))
}

#[tokio::test]
async fn vtx_013_a_stream_goes_to_stream_raw_predict_without_the_model() {
    let transport = Recorder::new([messages_text("hi")]);
    let (handle, events) = call(
        &claude(located("us-east5"), transport.clone(), None),
        hi(),
        labelled(),
    )
    .await;
    ok(&events);
    assert_eq!(handle.text(), "hi");
    let request = transport.request();
    assert_eq!(request.url, format!("{CLAUDE_URL}:streamRawPredict"));
    assert_eq!(
        header(&request, "authorization").as_deref(),
        Some("Bearer t1")
    );
    assert_eq!(header(&request, "x-api-key"), None);
    use base64::Engine as _;
    let labels = header(&request, "x-vertex-ai-labels").unwrap();
    assert_eq!(
        base64::engine::general_purpose::STANDARD
            .decode(labels)
            .unwrap(),
        br#"{"team":"ai"}"#.to_vec()
    );
    let body = transport.body();
    assert!(body.get("model").is_none(), "{body}");
    assert_eq!(body["anthropic_version"], "vertex-2023-10-16");
    assert_eq!(body["stream"], true);

    let transport = Recorder::new([messages_text("hi")]);
    let mut config = located("us-east5");
    config.headers = [("X-Vertex-AI-Labels".to_owned(), "mine".to_owned())].into();
    let (_, events) = call(&claude(config, transport.clone(), None), hi(), labelled()).await;
    ok(&events);
    assert_eq!(
        header(&transport.request(), "x-vertex-ai-labels").as_deref(),
        Some("mine")
    );
}

#[tokio::test]
async fn vtx_013_a_buffered_call_goes_to_raw_predict() {
    let transport = Recorder::new([Reply::json(json!({
        "id": "msg_1", "type": "message", "role": "assistant", "model": CLAUDE,
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn", "usage": {"input_tokens": 3, "output_tokens": 1},
    }))]);
    let message = suprnova_ai::generate(
        &claude(located("us-east5"), transport.clone(), None),
        hi(),
        StreamOptions::default(),
    )
    .await
    .unwrap();
    assert_eq!(message.text(), "hello");
    assert_eq!(message.usage.as_ref().unwrap().input_tokens, 3);
    assert_eq!(transport.request().url, format!("{CLAUDE_URL}:rawPredict"));
    let body = transport.body();
    assert!(
        body.get("stream").is_none() && body.get("model").is_none(),
        "{body}"
    );
    assert_eq!(body["anthropic_version"], "vertex-2023-10-16");
}

#[tokio::test]
async fn vtx_013_file_urls_are_downloaded_and_sent_as_base64() {
    let cat = vec![Message::new(
        MessageRole::User,
        vec![Part::File(
            FilePart::from_url("https://example.com/cat.png", Some("image/png")).unwrap(),
        )],
    )];
    let downloads = Recorder::new([Reply::bytes(vec![
        0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a,
    ])
    .header("content-type", "image/png")]);
    let transport = Recorder::new([messages_text("a cat")]);
    let (_, events) = call(
        &claude(located("us-east5"), transport.clone(), Some(downloads)),
        cat.clone(),
        StreamOptions::default(),
    )
    .await;
    ok(&events);
    assert_eq!(
        transport.body()["messages"][0]["content"][0],
        json!({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}})
    );
    let transport = Recorder::new([messages_text("a cat")]);
    let (_, events) = call(
        &claude(located("us-east5"), transport.clone(), None),
        cat,
        StreamOptions::default(),
    )
    .await;
    assert!(
        matches!(error(&events), Error::Validation(_)),
        "{:?}",
        error(&events)
    );
    assert_eq!(transport.count(), 0);
}

#[tokio::test]
async fn vtx_013_an_output_schema_uses_the_json_tool_and_tools_are_never_strict() {
    let schema = json!({"type": "object", "properties": {"a": {"type": "integer"}}});
    let transport = Recorder::new([messages_reply(&[
        json!({"type": "message_start", "message": {"id": "msg_1", "model": CLAUDE, "content": [], "stop_reason": null, "usage": {"input_tokens": 3, "output_tokens": 0}}}),
        json!({"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "tu1", "name": "json", "input": {}}}),
        json!({"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\"a\":1}"}}),
        json!({"type": "content_block_stop", "index": 0}),
        json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 5}}),
        json!({"type": "message_stop"}),
    ])]);
    let options = StreamOptions {
        output_schema: Some(schema.clone().try_into().unwrap()),
        tools: vec![tool("lookup").with_strict(true)],
        ..Default::default()
    };
    let (handle, events) = call(
        &claude(located("us-east5"), transport.clone(), None),
        hi(),
        options,
    )
    .await;
    ok(&events);
    assert_eq!(handle.text(), r#"{"a":1}"#);
    assert_eq!(handle.finish_reason(), Some("stop"));
    assert!(handle.message().tool_calls().is_empty());
    let body = transport.body();
    assert!(body.get("output_config").is_none(), "{body}");
    assert_eq!(
        body["tool_choice"],
        json!({"type": "any", "disable_parallel_tool_use": true})
    );
    let tools = body["tools"].as_array().unwrap();
    assert!(tools[0].get("strict").is_none(), "{}", tools[0]);
    assert_eq!(
        tools[1],
        json!({"name": "json", "description": "Respond with a JSON object.", "input_schema": schema})
    );
}

#[tokio::test]
async fn vtx_013_only_the_listed_anthropic_tools_are_taken() {
    let transport = Recorder::new([messages_text("ok")]);
    let options = StreamOptions {
        tools: vec![
            suprnova_ai::providers::anthropic::tools::web_search(Default::default()).unwrap(),
        ],
        ..Default::default()
    };
    let (_, events) = call(
        &claude(located("us-east5"), transport.clone(), None),
        hi(),
        options,
    )
    .await;
    assert!(
        matches!(error(&events), Error::Validation(_)),
        "{:?}",
        error(&events)
    );
    assert_eq!(transport.count(), 0);
    let options = StreamOptions {
        tools: vec![google_vertex::anthropic::tools::web_search_20250305(
            Some(2),
            None,
            None,
        )],
        ..Default::default()
    };
    let (_, events) = call(
        &claude(located("us-east5"), transport.clone(), None),
        hi(),
        options,
    )
    .await;
    ok(&events);
    assert_eq!(
        transport.body()["tools"][0],
        json!({"type": "web_search_20250305", "name": "web_search", "max_uses": 2})
    );
}

// VTX-014

fn chat_reply(chunks: &[Value]) -> Reply {
    let mut text: String = chunks.iter().map(|c| format!("data: {c}\n\n")).collect();
    text.push_str("data: [DONE]\n\n");
    Reply::bytes(text.into_bytes())
}

fn chat_text(text: &str) -> Reply {
    chat_reply(&[
        json!({"id": "r1", "model": "m", "choices": [{"index": 0, "delta": {"content": text}}]}),
        json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
    ])
}

fn maas(id: &str, config: ProviderConfig, transport: Arc<Recorder>) -> Model {
    let provider =
        google_vertex::maas::provider(config, transport).with_access_token_source(tokens("t1"));
    let mut registry = ProviderRegistry::new();
    google_vertex::maas::register_provider(&mut registry, provider).unwrap();
    let provider = registry.get_provider("@ai-sdk/google-vertex/maas").unwrap();
    assert_eq!(provider.name(), "google-vertex-maas");
    Model::new(id, provider)
}

#[tokio::test]
async fn vtx_014_maas_goes_to_the_openapi_endpoint_with_auth_last() {
    let transport = Recorder::new([chat_text("hi")]);
    let mut config = env(&[("GOOGLE_VERTEX_PROJECT", PROJECT)]);
    config.headers = [("Authorization".to_owned(), "Bearer mine".to_owned())].into();
    let (handle, events) = call(
        &maas("deepseek-ai/deepseek-v3.2-maas", config, transport.clone()),
        hi(),
        StreamOptions::default(),
    )
    .await;
    ok(&events);
    assert_eq!(handle.text(), "hi");
    let request = transport.request();
    assert_eq!(
        request.url,
        "https://aiplatform.googleapis.com/v1/projects/p1/locations/global/endpoints/openapi/chat/completions"
    );
    let values: Vec<&String> = request
        .headers
        .iter()
        .filter(|(n, _)| n.eq_ignore_ascii_case("authorization"))
        .map(|(_, v)| v)
        .collect();
    assert_eq!(values, vec!["Bearer t1"]);
    assert!(transport.body().get("max_tokens").is_none());

    let transport = Recorder::new([chat_text("hi")]);
    let (_, events) = call(
        &maas(
            "meta/llama-4-scout-17b-16e-instruct-maas",
            located("us-central1"),
            transport.clone(),
        ),
        hi(),
        StreamOptions::default(),
    )
    .await;
    ok(&events);
    assert_eq!(
        transport.request().url,
        "https://us-central1-aiplatform.googleapis.com/v1/projects/p1/locations/us-central1/endpoints/openapi/chat/completions"
    );
    assert_eq!(transport.body()["max_tokens"], 8192);

    let transport = Recorder::new([chat_text("hi")]);
    let limited = params(|p| {
        p.output = Some(OutputParams {
            max_tokens: Some(100),
            ..Default::default()
        })
    });
    let (_, events) = call(
        &maas(
            "meta/llama-4-maverick-17b-128e-instruct-maas",
            located("us-central1"),
            transport.clone(),
        ),
        hi(),
        limited,
    )
    .await;
    ok(&events);
    assert_eq!(transport.body()["max_tokens"], 100);

    let transport = Recorder::new([chat_text("hi")]);
    let (_, events) = call(
        &maas(
            "deepseek-ai/deepseek-v3.2-maas",
            located("us/x"),
            transport.clone(),
        ),
        hi(),
        StreamOptions::default(),
    )
    .await;
    configuration(error(&events));
    assert_eq!(transport.count(), 0);
}

// VTX-015

fn vertex_xai(config: ProviderConfig, transport: Arc<Recorder>) -> Model {
    let provider =
        google_vertex::xai::provider(config, transport).with_access_token_source(tokens("t1"));
    let mut registry = ProviderRegistry::new();
    google_vertex::xai::register_provider(&mut registry, provider).unwrap();
    let provider = registry.get_provider("@ai-sdk/google-vertex/xai").unwrap();
    assert_eq!(provider.name(), "google-vertex-xai");
    Model::new("xai/grok-4.20-reasoning", provider)
}

#[tokio::test]
async fn vtx_015_xai_goes_to_the_global_host_without_a_reasoning_effort() {
    let transport = Recorder::new([chat_text("hi")]);
    let options = StreamOptions {
        output_schema: Some(json!({"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"], "additionalProperties": false}).try_into().unwrap()),
        ..effort_options("high")
    };
    let (_, events) = call(
        &vertex_xai(located("us-central1"), transport.clone()),
        hi(),
        options,
    )
    .await;
    ok(&events);
    let request = transport.request();
    assert_eq!(
        request.url,
        "https://aiplatform.googleapis.com/v1/projects/p1/locations/us-central1/endpoints/openapi/chat/completions"
    );
    assert_eq!(
        header(&request, "authorization").as_deref(),
        Some("Bearer t1")
    );
    let body = transport.body();
    assert!(body.get("reasoning_effort").is_none(), "{body}");
    assert_eq!(body["stream_options"], json!({"include_usage": true}));
    assert_eq!(body["response_format"]["type"], "json_schema");

    let transport = Recorder::new([chat_text("hi")]);
    let (_, events) = call(
        &vertex_xai(
            env(&[("GOOGLE_VERTEX_PROJECT", PROJECT)]),
            transport.clone(),
        ),
        hi(),
        StreamOptions::default(),
    )
    .await;
    ok(&events);
    assert_eq!(
        transport.request().url,
        "https://aiplatform.googleapis.com/v1/projects/p1/locations/global/endpoints/openapi/chat/completions"
    );
}

fn effort_options(level: &str) -> StreamOptions {
    params(|p| {
        p.reasoning = ProviderValue::Value(ReasoningParams {
            effort: ProviderValue::Value(Some(level.into())),
        })
    })
}

#[tokio::test]
async fn vtx_015_only_image_urls_go_out_as_urls_and_usage_adds_reasoning() {
    let pdf = vec![Message::new(
        MessageRole::User,
        vec![Part::File(
            FilePart::from_url("https://example.com/a.pdf", Some("application/pdf")).unwrap(),
        )],
    )];
    let transport = Recorder::new([chat_text("hi")]);
    let (_, events) = call(
        &vertex_xai(located("global"), transport.clone()),
        pdf,
        StreamOptions::default(),
    )
    .await;
    assert!(
        matches!(error(&events), Error::Validation(_)),
        "{:?}",
        error(&events)
    );
    assert_eq!(transport.count(), 0);

    let image = vec![Message::new(
        MessageRole::User,
        vec![Part::File(
            FilePart::from_url("https://example.com/a.png", Some("image/png")).unwrap(),
        )],
    )];
    let transport = Recorder::new([chat_reply(&[
        json!({"id": "r1", "model": "m", "choices": [{"index": 0, "delta": {"content": "a cat"}}]}),
        json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        json!({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 6, "prompt_tokens_details": {"cached_tokens": 4}, "completion_tokens_details": {"reasoning_tokens": 3}}}),
    ])]);
    let (handle, events) = call(
        &vertex_xai(located("global"), transport.clone()),
        image,
        StreamOptions::default(),
    )
    .await;
    ok(&events);
    assert_eq!(
        transport.body()["messages"][0]["content"][0]["image_url"]["url"],
        "https://example.com/a.png"
    );
    let usage = handle.message().usage.as_ref().unwrap();
    assert_eq!(
        (
            usage.input_tokens,
            usage.cache_read_tokens,
            usage.output_tokens,
            usage.reasoning_tokens
        ),
        (10, Some(4), 9, Some(3))
    );
}

#[tokio::test]
async fn vtx_005_a_call_that_fails_validation_downloads_no_tool_result_file() {
    let downloads =
        Recorder::new([Reply::bytes(vec![1, 2, 3]).header("content-type", "image/png")]);
    let transport = Recorder::new([text_reply("ok")]);
    let model = with_downloads("gemini-2.5-flash", transport.clone(), downloads.clone());
    let (_, events) = call(
        &model,
        tool_result_history("https://example.com/shot.png", "image/png"),
        options_under("googleVertex", json!({"serviceTier": "flex"})),
    )
    .await;
    assert!(
        matches!(error(&events), Error::Validation(_)),
        "{:?}",
        error(&events)
    );
    assert_eq!(
        downloads.count(),
        0,
        "a call that fails validation downloaded a file"
    );
    assert_eq!(transport.count(), 0);
}
