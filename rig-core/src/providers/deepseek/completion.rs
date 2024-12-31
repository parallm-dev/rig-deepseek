use crate::completion::{
    CompletionError, CompletionModel as CompletionModelTrait, CompletionRequest,
    CompletionResponse as RigCompletionResponse, ModelChoice,
};
use crate::providers::deepseek::client::Client;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::iter;

pub const DEEPSEEK_API_BASE_URL: &str = "https://api.deepseek.com";
pub const DEEPSEEK_CHAT: &str = "deepseek-chat";
pub const DEEPSEEK_CODER: &str = "deepseek-coder";

#[derive(Debug, Deserialize, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub model: String,
    pub choices: Vec<ChoiceObject>,
    // Optionally include usage metrics if available
    // pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ChoiceObject {
    pub message: ChoiceMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    pub index: u32,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ChoiceMessage {
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolUse>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ToolUse {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize, Serialize)]
pub struct Usage {
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub total_tokens: i64,
}

impl TryFrom<CompletionResponse> for RigCompletionResponse<CompletionResponse> {
    type Error = CompletionError;

    fn try_from(resp: CompletionResponse) -> Result<Self, Self::Error> {
        let first_choice = resp
            .choices
            .get(0)
            .ok_or_else(|| CompletionError::ResponseError("No choices found".to_owned()))?;

        // Check for tool calls
        if let Some(tool_calls) = &first_choice.message.tool_calls {
            if let Some(tool_call) = tool_calls.get(0) {
                return Ok(RigCompletionResponse {
                    choice: ModelChoice::ToolCall(
                        tool_call.name.clone(),
                        tool_call.arguments.clone(),
                    ),
                    raw_response: resp,
                });
            }
        }

        // Check for message content
        if let Some(text) = &first_choice.message.content {
            return Ok(RigCompletionResponse {
                choice: ModelChoice::Message(text.clone()),
                raw_response: resp,
            });
        }

        Err(CompletionError::ResponseError(
            "No content or tool call in response".to_owned(),
        ))
    }
}

#[derive(Clone)]
pub struct CompletionModel {
    client: Client,
    model: String,
}

impl CompletionModel {
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }
}

#[async_trait]
impl CompletionModelTrait for CompletionModel {
    type Response = CompletionResponse;

    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<RigCompletionResponse<Self::Response>, CompletionError> {
        // Build JSON body using data from CompletionRequest
        let messages = request
            .chat_history
            .into_iter()
            .map(|m| json!({ "role": m.role, "content": m.content }))
            .chain(iter::once(json!({
                "role": "user",
                "content": request.prompt_with_context()
            })))
            .collect::<Vec<_>>();

        let max_tokens = request.max_tokens.unwrap_or(2048);

        let mut body = json!({
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        });

        if let Some(temp) = request.temperature {
            body["temperature"] = temp.into();
        }

        if !request.tools.is_empty() {
            // Include tools for function calling
            let ds_tools: Vec<_> = request.tools.into_iter().map(|tool| {
                json!({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    }
                })
            }).collect();
            body["tools"] = ds_tools.into();
        }

        if let Some(params) = &request.additional_params {
            crate::json_utils::merge_inplace(&mut body, params.clone());
        }

        // Make the API call
        let resp = self
            .client
            .post("/chat/completions")
            .json(&body)
            .send()
            .await
            .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

        // Handle the response
        if resp.status().is_success() {
            let ds_resp = resp
                .json::<ApiResponse<CompletionResponse>>()
                .await
                .map_err(|e| CompletionError::ResponseError(e.to_string()))?;

            match ds_resp {
                ApiResponse::Success(data) => data.try_into(), // Uses the TryFrom implementation
                ApiResponse::Error(error) => Err(CompletionError::ProviderError(error.message)),
            }
        } else {
            let error_text = resp.text().await.unwrap_or_default();
            Err(CompletionError::ProviderError(error_text))
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ApiResponse<T> {
    Success(T),
    Error(ApiErrorResponse),
}

#[derive(Debug, Deserialize)]
pub struct ApiErrorResponse {
    pub message: String,
}
