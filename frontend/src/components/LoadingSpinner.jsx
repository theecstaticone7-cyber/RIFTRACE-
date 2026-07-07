import { useEffect, useState } from 'react'
import { Loader2 } from 'lucide-react'

// The free-tier backend spins down after inactivity and takes 30-60s to
// wake on the next request. Past this threshold a request would normally
// already be done, so it's a good point to reassure the user it's not stuck.
const COLD_START_HINT_MS = 5000

export default function LoadingSpinner({ label = 'Loading...' }) {
  const [showColdStartHint, setShowColdStartHint] = useState(false)

  useEffect(() => {
    setShowColdStartHint(false)
    const timer = setTimeout(() => setShowColdStartHint(true), COLD_START_HINT_MS)
    return () => clearTimeout(timer)
  }, [label])

  return (
    <div className="flex items-center gap-2 text-sm text-neutral-400">
      <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin" />
      <span>
        {label}
        {showColdStartHint && (
          <span className="block text-neutral-500">
            Still working -- the backend is on a free tier and may be waking up from sleep
            (can take up to a minute).
          </span>
        )}
      </span>
    </div>
  )
}
