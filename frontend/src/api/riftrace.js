// Thin wrappers around the RiftRace backend's endpoints. Each returns a
// Promise that resolves with the parsed JSON body, or rejects with an
// ApiError (see client.js) carrying a user-friendly message.
import apiFetch from './client'

export function predictTransaction(txId) {
  return apiFetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ tx_id: Number(txId) }),
  })
}

export function getTransactionGraph(txId) {
  return apiFetch(`/transaction/${txId}/graph`)
}

export function investigateTransaction(txId) {
  return apiFetch(`/investigate/${txId}`, { method: 'POST' })
}

export function getStats() {
  return apiFetch('/stats')
}

export function checkHealth() {
  return apiFetch('/health')
}

export function getFlaggedTransactions(limit = 20) {
  return apiFetch(`/flagged?limit=${limit}`)
}
