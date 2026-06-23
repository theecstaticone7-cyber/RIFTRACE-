import { Loader2, Wifi, WifiOff } from 'lucide-react'

// A small abstract fault-line mark (the "rift") rather than a generic
// gradient box -- two crossing fractured strokes in the accent color.
function LogoMark() {
  return (
    <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-neutral-800 bg-neutral-800/60 text-accent-hover">
      <svg viewBox="0 0 24 24" width="18" height="18" fill="none" aria-hidden="true">
        <path
          d="M6 19L11 13L9 9L17 5"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <path
          d="M7 5L13 11L11 15L18 19"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          opacity="0.45"
        />
      </svg>
    </span>
  )
}

// `isOnline` is null while the initial health check is in flight, then
// true/false once it resolves -- drives the status indicator on the right.
export default function Header({ isOnline }) {
  const StatusIcon = isOnline === null ? Loader2 : isOnline ? Wifi : WifiOff

  return (
    <header className="mb-4 flex flex-wrap items-center justify-between gap-3 border-b border-neutral-800 pb-3">
      <div className="flex items-center gap-2.5">
        <LogoMark />
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-neutral-100">RiftRace</h1>
          <p className="text-xs text-neutral-500">Bitcoin transaction risk analysis</p>
        </div>
      </div>

      <div
        className={`flex items-center gap-1.5 text-xs font-medium ${
          isOnline === null ? 'text-neutral-400' : isOnline ? 'text-emerald-400' : 'text-red-400'
        }`}
      >
        <StatusIcon className={`h-3.5 w-3.5 ${isOnline === null ? 'animate-spin' : ''}`} />
        {isOnline === null ? 'Checking backend' : isOnline ? 'Online' : 'Backend unreachable'}
      </div>
    </header>
  )
}
