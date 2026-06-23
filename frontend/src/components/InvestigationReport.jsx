import { BookOpen, ChevronRight, ClipboardList, Info, ScanSearch, Sparkles, Workflow } from 'lucide-react'
import { getRiskTier } from '../lib/risk'
import LoadingSpinner from './LoadingSpinner'
import ErrorBanner from './ErrorBanner'

const STAGES = ['Analyze', 'Retrieve', 'Reason', 'Recommend']

// A static stepper, not a progress indicator -- it labels the four-agent
// pipeline that produced this report, so it reads as a structured
// multi-step analysis rather than a single LLM reply.
function PipelineStepper() {
  return (
    <div className="flex items-center gap-1 text-[10px] font-medium uppercase tracking-wide text-neutral-500">
      {STAGES.map((stage, i) => (
        <span key={stage} className="flex items-center gap-1">
          <span className="rounded-sm bg-neutral-800 px-1.5 py-0.5">{stage}</span>
          {i < STAGES.length - 1 && <ChevronRight className="h-3 w-3 text-neutral-700" />}
        </span>
      ))}
    </div>
  )
}

// Deliberately a muted info tag, not ErrorBanner's red/amber -- REASON/
// RECOMMEND falling back to a clear message (no GROQ_API_KEY, or a failed
// call) is an expected state, not a failure.
function GroundedTag({ grounded }) {
  if (grounded) {
    return (
      <span className="inline-flex w-fit items-center gap-1 rounded-sm bg-neutral-800 px-1.5 py-0.5 text-[11px] font-semibold uppercase tracking-wide text-neutral-300">
        <Sparkles className="h-3 w-3" />
        Reasoning grounded by LLM
      </span>
    )
  }
  return (
    <span className="inline-flex w-fit items-center gap-1 rounded-sm bg-neutral-800/60 px-1.5 py-0.5 text-[11px] font-medium text-neutral-500">
      <Info className="h-3 w-3" />
      LLM reasoning unavailable
    </span>
  )
}

function ReportTile({ icon: Icon, title, accentBorder, badge, children }) {
  return (
    <div className={`tile flex flex-col gap-1.5 p-3 ${accentBorder ? `border-l-2 ${accentBorder}` : ''}`}>
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-neutral-400">
          <Icon className="h-3.5 w-3.5" />
          {title}
        </div>
        {badge}
      </div>
      {children}
    </div>
  )
}

// The four ANALYZE/RETRIEVE/REASON/RECOMMEND outputs laid out as their own
// tiles in a 2x2 grid -- meant to read as a structured report, not a chat
// reply. Risk assessment reuses the same risk-tier coloring as
// PredictionResult, since severity is real signal here too, not decoration.
export default function InvestigationReport({ result, loading, error }) {
  const tier = result && getRiskTier(result.probability_illicit)

  return (
    <div className="panel flex flex-col gap-3 border-neutral-700 p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="flex items-center gap-1.5 text-sm font-semibold text-neutral-100">
          <Workflow className="h-4 w-4 text-neutral-500" />
          Investigation report
        </h3>
        <PipelineStepper />
      </div>

      {loading && (
        <LoadingSpinner label="Running multi-agent investigation... this runs several LLM calls and can take 10-20s" />
      )}

      <ErrorBanner error={error} />

      {!loading && !error && result && (
        <div className="flex flex-col gap-3">
          <GroundedTag grounded={result.grounded} />

          <div className="grid gap-3 sm:grid-cols-2">
            <ReportTile icon={ScanSearch} title="Analysis">
              <p className="text-sm leading-relaxed text-neutral-300">{result.analysis}</p>
            </ReportTile>

            <ReportTile icon={BookOpen} title="Retrieved AML knowledge">
              <div className="flex flex-wrap gap-1.5">
                {result.retrieved_sources.map((source) => (
                  <span
                    key={source}
                    className="rounded-sm bg-neutral-700/70 px-1.5 py-0.5 text-[11px] text-neutral-200"
                  >
                    {source}
                  </span>
                ))}
              </div>
            </ReportTile>

            <ReportTile
              icon={tier.icon}
              title="Risk assessment"
              accentBorder={tier.accentBorder}
              badge={
                <span
                  className={`inline-flex shrink-0 items-center gap-1 rounded-sm px-1.5 py-0.5 text-[11px] font-semibold uppercase tracking-wide ${tier.tag}`}
                >
                  {tier.label} risk
                </span>
              }
            >
              <p className="text-sm leading-relaxed text-neutral-300">{result.risk_assessment}</p>
            </ReportTile>

            <ReportTile icon={ClipboardList} title="Recommended actions">
              <p className="text-sm leading-relaxed text-neutral-300">{result.recommended_actions}</p>
            </ReportTile>
          </div>
        </div>
      )}
    </div>
  )
}
