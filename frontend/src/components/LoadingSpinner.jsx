import { Loader2 } from 'lucide-react'

export default function LoadingSpinner({ label = 'Loading...' }) {
  return (
    <div className="flex items-center gap-2 text-sm text-neutral-400">
      <Loader2 className="h-3.5 w-3.5 animate-spin" />
      {label}
    </div>
  )
}
