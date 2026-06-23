// Shared between NeighborsList's table and MoneyFlowGraph's canvas so the
// two views of the same transaction graph never disagree on how many
// neighbors actually render -- this is the one place that cap is defined.
export const MAX_RENDERED_NEIGHBORS = 50

// The ground-truth class colors used anywhere a tx_id's known_class is
// drawn directly (the graph canvas, the dataset donut chart). Risk *tier*
// (predicted probability) has its own red/orange/amber/neutral/emerald
// scale in lib/risk.js -- this is the simpler 3-way ground truth.
export const CLASS_COLOR_HEX = {
  illicit: '#ef4444',
  licit: '#10b981',
  unknown: '#737373',
}
