import { useEffect, useRef, useState } from 'react'
import Header from './components/Header'
import StatsPanel from './components/StatsPanel'
import FlaggedList from './components/FlaggedList'
import TransactionLookup from './components/TransactionLookup'
import ErrorBanner from './components/ErrorBanner'
import { checkHealth } from './api/riftrace'

function App() {
  // isOnline drives the header status pill (null while checking); the
  // detailed error banner only shows up once we know it's actually down.
  const [isOnline, setIsOnline] = useState(null)
  const [backendError, setBackendError] = useState(null)

  // `key` makes the object reference change on every click (even re-clicking
  // the same tx_id), so TransactionLookup's effect always re-fires.
  const [autoLookup, setAutoLookup] = useState(null)
  const lookupSectionRef = useRef(null)

  useEffect(() => {
    checkHealth()
      .then(() => setIsOnline(true))
      .catch((err) => {
        setIsOnline(false)
        setBackendError(err)
      })
  }, [])

  function handleFlaggedSelect(txId) {
    setAutoLookup({ txId, key: Date.now() })
    lookupSectionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-300">
      <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6">
        <Header isOnline={isOnline} />

        {backendError && (
          <div className="mb-3">
            <ErrorBanner error={backendError} />
          </div>
        )}

        <div className="space-y-4">
          <StatsPanel />

          <div className="grid gap-4 lg:grid-cols-[420px_1fr]">
            <FlaggedList onSelect={handleFlaggedSelect} selectedTxId={autoLookup?.txId} />
            <div ref={lookupSectionRef}>
              <TransactionLookup autoLookup={autoLookup} />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
