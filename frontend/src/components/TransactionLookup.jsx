import { useCallback, useEffect, useRef, useState } from 'react'
import { Hash, Loader2, Search } from 'lucide-react'
import { predictTransaction, getTransactionGraph, investigateTransaction } from '../api/riftrace'
import PredictionResult from './PredictionResult'
import NeighborsList from './NeighborsList'
import MoneyFlowGraph from './MoneyFlowGraph'
import InvestigationReport from './InvestigationReport'
import ErrorBanner from './ErrorBanner'
import LoadingSpinner from './LoadingSpinner'

// `autoLookup` is `{ txId, key }` set by a click on a Top Flagged Transactions
// row elsewhere on the page. `key` changes on every click (even re-clicking
// the same tx_id) so the effect below always re-runs the lookup.
export default function TransactionLookup({ autoLookup }) {
  const [txId, setTxId] = useState('')
  // Tracks which autoLookup.key has already been synced into `txId`, so the
  // sync below runs during render (React's documented pattern for adjusting
  // state from a changed prop) instead of as a setState call inside an
  // effect body, which React flags as a cascading-render risk.
  const [syncedKey, setSyncedKey] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [graph, setGraph] = useState(null)
  const [investigation, setInvestigation] = useState(null)
  const [predictionError, setPredictionError] = useState(null)
  const [graphError, setGraphError] = useState(null)
  const [investigationError, setInvestigationError] = useState(null)
  const [loading, setLoading] = useState(false)
  // /investigate runs four agent steps including two LLM calls, so it's much
  // slower than /predict + /transaction/.../graph -- it gets its own loading
  // flag instead of sharing `loading`, so the prediction and graph render as
  // soon as they're ready, with only the investigation report still spinning.
  const [investigationLoading, setInvestigationLoading] = useState(false)
  // Guards against a slow /investigate call from a previous lookup resolving
  // after a newer one has already started and clobbering its state.
  const investigateRequestRef = useRef(0)
  // Distinguishes "never searched yet" (show the empty-state placeholder)
  // from "searched but both calls failed" (show the error banners instead).
  const [hasSearched, setHasSearched] = useState(false)

  const runLookup = useCallback(async (rawTxId) => {
    const trimmed = String(rawTxId).trim()
    if (!trimmed) return

    setHasSearched(true)
    setLoading(true)
    setPrediction(null)
    setGraph(null)
    setInvestigation(null)
    setPredictionError(null)
    setGraphError(null)
    setInvestigationError(null)
    setInvestigationLoading(true)

    // Fired separately from the allSettled batch below (not awaited here) so
    // the prediction/graph panels can render as soon as they're ready instead
    // of waiting on the much slower multi-agent investigation. requestId
    // guards against this resolving after a newer lookup has already started.
    const investigateRequestId = ++investigateRequestRef.current
    investigateTransaction(trimmed)
      .then((result) => {
        if (investigateRequestRef.current === investigateRequestId) setInvestigation(result)
      })
      .catch((err) => {
        if (investigateRequestRef.current === investigateRequestId) setInvestigationError(err)
      })
      .finally(() => {
        if (investigateRequestRef.current === investigateRequestId) setInvestigationLoading(false)
      })

    // /predict and /transaction/{id}/graph are independent reads about the
    // same tx_id. allSettled (not all) means one failing -- e.g. the graph
    // endpoint 500ing -- doesn't blank out an otherwise-successful prediction.
    const [predictionOutcome, graphOutcome] = await Promise.allSettled([
      predictTransaction(trimmed),
      getTransactionGraph(trimmed),
    ])

    if (predictionOutcome.status === 'fulfilled') {
      setPrediction(predictionOutcome.value)
    } else {
      setPredictionError(predictionOutcome.reason)
    }

    if (graphOutcome.status === 'fulfilled') {
      setGraph(graphOutcome.value)
    } else {
      setGraphError(graphOutcome.reason)
    }

    setLoading(false)
  }, [])

  // Fills the input when a new flagged-transaction click comes in. Adjusting
  // state during render (not inside an effect) avoids an extra render pass.
  if (autoLookup && autoLookup.key !== syncedKey) {
    setSyncedKey(autoLookup.key)
    setTxId(String(autoLookup.txId))
  }

  // Running the actual lookup is a real side effect (a network call), so
  // that part belongs in an effect, separate from the render-time sync above.
  // Deferred via .then() so none of runLookup's setState calls -- even the
  // ones before its first internal await -- run synchronously inside this
  // effect body, which React's lint rules disallow.
  useEffect(() => {
    if (!autoLookup) return
    Promise.resolve().then(() => runLookup(autoLookup.txId))
  }, [autoLookup, runLookup])

  function handleSubmit(event) {
    event.preventDefault()
    runLookup(txId)
  }

  // Clicking a neighbor node in the money-flow graph re-centers the whole
  // panel on it: the input reflects the new tx and a fresh lookup runs for
  // it, same as if the user had typed/clicked it themselves.
  function handleSelectNeighbor(neighborTxId) {
    setTxId(neighborTxId)
    runLookup(neighborTxId)
  }

  const showPlaceholder = !hasSearched && !loading

  return (
    <section className="panel flex h-full flex-col p-4">
      <h2 className="flex items-center gap-2 text-sm font-semibold text-neutral-100">
        <Search className="h-4 w-4 text-neutral-500" />
        Investigate transaction
      </h2>
      <p className="mb-2 text-xs text-neutral-500">
        Click a flagged transaction on the left, or enter a tx_id directly below.
      </p>

      <form onSubmit={handleSubmit} className="flex flex-col gap-2 sm:flex-row">
        <div className="relative flex-1">
          <Hash className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-neutral-600" />
          <input
            type="text"
            inputMode="numeric"
            value={txId}
            onChange={(event) => setTxId(event.target.value)}
            placeholder="e.g. 232629023"
            className="w-full rounded-md border border-neutral-700 bg-neutral-950 py-1.5 pl-8 pr-2.5 font-mono text-sm text-neutral-100 placeholder:text-neutral-600 focus:border-accent focus:outline-none"
          />
        </div>
        <button
          type="submit"
          disabled={loading || !txId.trim()}
          className="flex items-center justify-center gap-1.5 rounded-md bg-accent px-4 py-1.5 text-sm font-medium text-neutral-100 hover:bg-accent-hover disabled:cursor-not-allowed disabled:bg-neutral-800 disabled:text-neutral-500"
        >
          {loading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Search className="h-3.5 w-3.5" />}
          {loading ? 'Checking...' : 'Check transaction'}
        </button>
      </form>

      {loading && (
        <div className="mt-3">
          <LoadingSpinner label="Querying RiftRace backend..." />
        </div>
      )}

      {showPlaceholder && (
        <div className="mt-3 flex flex-1 flex-col items-center justify-center gap-3 rounded-md border border-dashed border-neutral-800 px-6 py-12 text-center">
          <svg
            width="56"
            height="56"
            viewBox="0 0 56 56"
            fill="none"
            className="text-neutral-700"
            aria-hidden="true"
          >
            <circle cx="14" cy="14" r="4" stroke="currentColor" strokeWidth="1.5" />
            <circle cx="42" cy="12" r="3" stroke="currentColor" strokeWidth="1.5" />
            <circle cx="28" cy="30" r="5" stroke="currentColor" strokeWidth="1.5" />
            <circle cx="10" cy="44" r="3" stroke="currentColor" strokeWidth="1.5" />
            <circle cx="44" cy="42" r="4" stroke="currentColor" strokeWidth="1.5" />
            <path
              d="M17 16.5L24 26M32 27.5L39 15M24 33L15 41.5M33 33L40 39"
              stroke="currentColor"
              strokeWidth="1.5"
            />
          </svg>
          <div className="max-w-xs">
            <p className="text-sm font-medium text-neutral-300">No transaction selected</p>
            <p className="mt-1 text-xs text-neutral-500">
              Click a row in Top flagged transactions, or enter a tx_id above, to see its risk
              prediction and transaction graph here.
            </p>
          </div>
        </div>
      )}

      {hasSearched && !loading && (
        <div className="mt-3 flex flex-1 flex-col gap-3">
          <div className="grid flex-1 gap-3 sm:grid-cols-[260px_1fr]">
            <div className="flex flex-col gap-2">
              <ErrorBanner error={predictionError} />
              {prediction && (
                <div className="flex-1">
                  <PredictionResult result={prediction} />
                </div>
              )}
            </div>
            <div className="flex flex-col gap-2">
              <ErrorBanner error={graphError} />
              {graph && (
                <div className="min-h-[240px] flex-1">
                  <MoneyFlowGraph
                    graph={graph}
                    prediction={prediction}
                    onSelectNeighbor={handleSelectNeighbor}
                  />
                </div>
              )}
            </div>
          </div>

          {prediction && (
            <InvestigationReport
              result={investigation}
              loading={investigationLoading}
              error={investigationError}
            />
          )}

          {graph && <NeighborsList graph={graph} />}
        </div>
      )}
    </section>
  )
}
