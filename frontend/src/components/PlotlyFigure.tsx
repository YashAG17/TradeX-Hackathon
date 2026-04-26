import Plot from "react-plotly.js";
import type { PlotlyFigureJSON } from "../api/types";

interface PlotlyFigureProps {
  figure: PlotlyFigureJSON;
  height?: number;
  className?: string;
}

// Thin wrapper around react-plotly.js. The backend produces fully-styled figures
// (palette mirrored from dashboard.py) so this component just hands them off.
export function PlotlyFigure({ figure, height, className }: PlotlyFigureProps) {
  const layout = {
    ...figure.layout,
    autosize: true,
    ...(height ? { height } : {}),
  };

  return (
    <div className={className}>
      <Plot
        data={figure.data as Plotly.Data[]}
        layout={layout as Plotly.Layout}
        config={{ displayModeBar: false, responsive: true }}
        useResizeHandler
        style={{ width: "100%", height: height ? `${height}px` : "100%" }}
      />
    </div>
  );
}
