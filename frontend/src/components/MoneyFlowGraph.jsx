import { useEffect, useMemo, useRef, useState } from 'react'
import ForceGraph2D from 'react-force-graph-2d'
import { Waypoints } from 'lucide-react'
import { CLASS_COLOR_HEX, MAX_RENDERED_NEIGHBORS } from '../lib/graph'

const ACCENT_HEX = '#6a6ea0'

// force-graph needs explicit pixel dimensions -- it doesn't size itself to
// a flexbox/grid parent -- so we measure the wrapping div and feed that in.
// Tracking height too (not just width) means the canvas fills exactly the
// space the flex layout gives it instead of leaving a void below a fixed
// height or overflowing a smaller one.
function useElementSize() {
  const ref = useRef(null)
  const [size, setSize] = useState({ width: 0, height: 0 })

  useEffect(() => {
    if (!ref.current) return
    const observer = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect
      setSize({ width, height })
    })
    observer.observe(ref.current)
    return () => observer.disconnect()
  }, [])

  return [ref, size]
}

// Builds a small ego-network: the looked-up tx in the center, its (capped)
// neighbors around it, edges pointing in the direction money actually moved.
function buildGraphData(graph) {
  const centerId = String(graph.tx_id)
  const shown = graph.neighbors.slice(0, MAX_RENDERED_NEIGHBORS)

  const nodes = [
    // fx/fy pin the center at the simulation's origin so neighbors radiate
    // outward from a fixed point instead of the whole layout drifting.
    { id: centerId, knownClass: graph.known_class, isCenter: true, fx: 0, fy: 0 },
    ...shown.map((n) => ({ id: String(n.tx_id), knownClass: n.known_class, isCenter: false })),
  ]

  // direction is from the *looked-up* tx's point of view: "outgoing" means
  // money left it for the neighbor, "incoming" means the neighbor sent it.
  const links = shown.map((n) => ({
    source: n.direction === 'outgoing' ? centerId : String(n.tx_id),
    target: n.direction === 'outgoing' ? String(n.tx_id) : centerId,
  }))

  return { nodes, links, shownCount: shown.length }
}

export default function MoneyFlowGraph({ graph, prediction, onSelectNeighbor }) {
  const [containerRef, { width, height }] = useElementSize()
  const fgRef = useRef(null)
  const [hovered, setHovered] = useState(null)

  const { nodes, links, shownCount } = useMemo(() => buildGraphData(graph), [graph])
  const hiddenCount = graph.num_neighbors - shownCount

  // Re-frame on every new lookup (including re-centering onto a clicked
  // neighbor) once the simulation has settled instead of fighting it mid-tick.
  function handleEngineStop() {
    fgRef.current?.zoomToFit(400, 36)
  }

  // The default link distance/charge are tuned for bigger graphs and leave
  // a 1-2 neighbor ego-network visually overlapping. Widen both once, when
  // the canvas instance first mounts -- the config applies for the engine's
  // whole lifetime, not just the dataset active at mount time.
  function handleGraphRef(instance) {
    fgRef.current = instance
    if (instance) {
      instance.d3Force('link').distance(90)
      instance.d3Force('charge').strength(-100)
    }
  }

  if (graph.num_neighbors === 0) {
    return (
      <div className="tile flex h-full min-h-[200px] flex-col items-center justify-center gap-2 px-6 text-center">
        <Waypoints className="h-5 w-5 text-neutral-600" />
        <p className="text-xs text-neutral-500">No connected transactions in the graph.</p>
      </div>
    )
  }

  return (
    <div className="tile flex h-full flex-col overflow-hidden">
      <div className="flex items-center justify-between border-b border-neutral-800 px-3 py-1.5">
        <span className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wide text-neutral-400">
          <Waypoints className="h-3 w-3" />
          Money flow
        </span>
        <span className="font-mono text-[11px] text-neutral-500">
          {hovered
            ? `tx ${hovered.id} -- ${
                hovered.isCenter && prediction
                  ? `${(prediction.probability_illicit * 100).toFixed(1)}% illicit`
                  : hovered.knownClass
              }`
            : hiddenCount > 0
              ? `showing ${shownCount} of ${graph.num_neighbors}`
              : `${shownCount} neighbor${shownCount === 1 ? '' : 's'}`}
        </span>
      </div>

      <div ref={containerRef} className="min-h-0 flex-1">
        {width > 0 && height > 0 && (
          <ForceGraph2D
            ref={handleGraphRef}
            graphData={{ nodes, links }}
            width={width}
            height={height}
            backgroundColor="rgba(0,0,0,0)"
            nodeId="id"
            nodeVal={(node) => (node.isCenter ? 22 : 5)}
            nodeColor={(node) => CLASS_COLOR_HEX[node.knownClass] || CLASS_COLOR_HEX.unknown}
            nodeLabel={() => ''}
            linkColor={() => 'rgba(255,255,255,0.16)'}
            linkWidth={1}
            linkDirectionalArrowLength={5}
            linkDirectionalArrowRelPos={1}
            linkDirectionalArrowColor={() => 'rgba(255,255,255,0.4)'}
            cooldownTicks={80}
            onEngineStop={handleEngineStop}
            onNodeHover={(node) => {
              setHovered(node)
              if (containerRef.current) {
                containerRef.current.style.cursor = node ? 'pointer' : 'default'
              }
            }}
            onNodeClick={(node) => {
              if (!node.isCenter) onSelectNeighbor(node.id)
            }}
            nodeCanvasObjectMode={() => 'after'}
            nodeCanvasObject={(node, ctx, globalScale) => {
              // A ring around the center node -- same accent used for
              // selection/focus everywhere else -- so it reads as "you are
              // here" without needing a different fill color.
              if (!node.isCenter) return
              ctx.beginPath()
              ctx.arc(node.x, node.y, 10.5, 0, 2 * Math.PI)
              ctx.strokeStyle = ACCENT_HEX
              ctx.lineWidth = 1.5 / globalScale
              ctx.stroke()
            }}
          />
        )}
      </div>

      {hiddenCount > 0 && (
        <p className="border-t border-neutral-800 px-3 py-1.5 text-[11px] text-neutral-500">
          Showing {shownCount} of {graph.num_neighbors} neighbors ({hiddenCount} more not shown).
        </p>
      )}
    </div>
  )
}
