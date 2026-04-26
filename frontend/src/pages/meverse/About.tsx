export function About() {
  return (
    <div className="card prose prose-invert max-w-none animate-fade-up">
      <h2 className="text-xl font-semibold text-accent mb-3">About MEVerse</h2>
      <p className="text-sm text-muted leading-relaxed">
        <strong className="text-text">MEVerse</strong> is a bot-aware market
        surveillance benchmark built on a simulated AMM (Automated Market Maker)
        environment.
      </p>

      <h3 className="text-base font-semibold mt-6 mb-2">Tasks</h3>
      <ul className="text-sm space-y-2">
        <li>
          <strong className="text-text">Burst Detection (Easy)</strong> — sudden,
          acute spikes from a bot hammering the pool. Primary signal:{" "}
          <code className="text-text">burst_indicator</code>. Threshold rules are
          usually enough.
        </li>
        <li>
          <strong className="text-text">Pattern Manipulation (Medium)</strong>{" "}
          — sustained, rhythmic coordination — repeating timing intervals or
          size signatures. Bots can have low burst but high pattern.
        </li>
        <li>
          <strong className="text-text">Full Market Surveillance (Hard)</strong>{" "}
          — both threats at once, mixed with normal traffic. 60 steps,
          mid-range initial bot confidence.
        </li>
      </ul>

      <h3 className="text-base font-semibold mt-6 mb-2">Actions</h3>
      <ul className="text-sm space-y-1 list-disc pl-5">
        <li><strong>ALLOW</strong> — let it through</li>
        <li><strong>MONITOR</strong> — watch more closely</li>
        <li><strong>FLAG</strong> — mark as suspicious for review</li>
        <li><strong>BLOCK</strong> — block the activity</li>
      </ul>

      <h3 className="text-base font-semibold mt-6 mb-2">Scoring</h3>
      <div className="overflow-auto rounded-lg border border-border my-3">
        <table className="w-full text-sm">
          <thead className="bg-bg">
            <tr>
              <th className="text-left text-xs font-semibold uppercase tracking-wide text-accent px-3 py-2 border-b border-border">Component</th>
              <th className="text-left text-xs font-semibold uppercase tracking-wide text-accent px-3 py-2 border-b border-border">Weight</th>
              <th className="text-left text-xs font-semibold uppercase tracking-wide text-accent px-3 py-2 border-b border-border">Measures</th>
            </tr>
          </thead>
          <tbody>
            {[
              ["Detection", "50%", "Correct identification of suspicious activity"],
              ["False Positive", "20%", "Avoiding flagging normal activity"],
              ["False Negative", "15%", "Avoiding missing suspicious activity"],
              ["Health", "10%", "Preserving healthy market behavior"],
              ["Overblocking", "5%", "Not over-blocking normal users"],
            ].map((row) => (
              <tr key={row[0]} className="hover:bg-accent/5">
                {row.map((cell, i) => (
                  <td key={i} className="px-3 py-2 border-b border-border text-sm">{cell}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h3 className="text-base font-semibold mt-6 mb-2">AMM Dynamics</h3>
      <p className="text-sm text-muted leading-relaxed">
        The environment uses a constant-product AMM (<code className="text-text">x · y = k</code>). Agent
        actions affect AMM state — blocking suspicious activity reduces bot
        confidence and volatility, while allowing it increases both. Early
        decisions shape future observations.
      </p>
    </div>
  );
}
