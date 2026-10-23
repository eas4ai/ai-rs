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

//! Batches: many text or image requests that the provider runs on its own
//! time, addressed afterwards by the batch id.

use super::OperationOptions;
use crate::{Error, Provider, models::call::with_retries, providers::operations::*};

/// Start a batch of requests and return its id, its status and the warnings
/// about its requests.
pub async fn start(
    provider: &dyn Provider,
    requests: Vec<BatchRequest>,
    params: BatchParams,
    options: OperationOptions,
) -> Result<BatchStarted, Error> {
    with_retries(&options.retry, || {
        provider.batch_start(requests.clone(), params.clone())
    })
    .await
}

pub async fn status(
    provider: &dyn Provider,
    batch_id: &str,
    options: OperationOptions,
) -> Result<BatchStatus, Error> {
    with_retries(&options.retry, || provider.batch_status(batch_id)).await
}

pub async fn cancel(
    provider: &dyn Provider,
    batch_id: &str,
    options: OperationOptions,
) -> Result<(), Error> {
    with_retries(&options.retry, || provider.batch_cancel(batch_id)).await
}

/// One page of batches, `page_size` long at most, from `cursor` on.
pub async fn list(
    provider: &dyn Provider,
    page_size: Option<u32>,
    cursor: Option<String>,
    options: OperationOptions,
) -> Result<BatchPage, Error> {
    with_retries(&options.retry, || {
        provider.batch_list(page_size, cursor.clone())
    })
    .await
}

/// Stream one result per request of a finished batch, in the provider's
/// order. Opening the stream retries; an error inside it ends it.
pub async fn results(
    provider: &dyn Provider,
    batch_id: &str,
    options: OperationOptions,
) -> Result<BatchResults, Error> {
    use futures::StreamExt;
    let results = with_retries(&options.retry, || provider.batch_results(batch_id)).await?;
    Ok(results.take(4).boxed())
}
