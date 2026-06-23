import { WifiOff, SearchX, ServerCrash, AlertCircle } from 'lucide-react'

// Renders nothing if there's no error, so callers can always include this
// unconditionally instead of needing their own `{error && ...}` checks.
const STYLES_BY_KIND = {
  network: 'border-amber-800/50 bg-amber-950/30 text-amber-300',
  not_found: 'border-neutral-700 bg-neutral-800/60 text-neutral-300',
  server: 'border-red-800/50 bg-red-950/30 text-red-300',
  client: 'border-red-800/50 bg-red-950/30 text-red-300',
}

const LABEL_BY_KIND = {
  network: 'Connection problem',
  not_found: 'Not found',
  server: 'Server error',
  client: 'Request error',
}

const ICON_BY_KIND = {
  network: WifiOff,
  not_found: SearchX,
  server: ServerCrash,
  client: AlertCircle,
}

export default function ErrorBanner({ error }) {
  if (!error) return null

  const kind = error.kind || 'client'
  const Icon = ICON_BY_KIND[kind]

  return (
    <div className={`flex gap-2 rounded-md border px-3 py-2 text-sm ${STYLES_BY_KIND[kind]}`}>
      <Icon className="mt-0.5 h-4 w-4 shrink-0" />
      <div>
        <p className="text-xs font-semibold uppercase tracking-wide">{LABEL_BY_KIND[kind]}</p>
        <p className="mt-1">{error.message}</p>
      </div>
    </div>
  )
}
