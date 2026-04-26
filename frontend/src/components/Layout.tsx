import { NavLink, Outlet } from "react-router-dom";

const navLinkClass = ({ isActive }: { isActive: boolean }) =>
  [
    "px-4 py-2 rounded-md text-sm font-medium transition-colors",
    isActive ? "bg-accent text-bg" : "text-muted hover:text-text hover:bg-border/40",
  ].join(" ");

export function Layout() {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-border bg-surface/80 backdrop-blur sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <NavLink to="/" className="flex items-center gap-3">
            <span className="w-8 h-8 rounded-md bg-accent text-bg font-bold text-lg grid place-items-center">
              T
            </span>
            <div className="leading-tight">
              <div className="font-semibold tracking-tight">TradeX + MEVerse</div>
              <div className="text-muted text-xs">Surveillance Dashboard</div>
            </div>
          </NavLink>
          <nav className="flex gap-2">
            <NavLink to="/" end className={navLinkClass}>
              Home
            </NavLink>
            <NavLink to="/meverse" className={navLinkClass}>
              MEVerse
            </NavLink>
            <NavLink to="/tradex" className={navLinkClass}>
              TradeX
            </NavLink>
          </nav>
        </div>
      </header>

      <main className="flex-1 max-w-7xl w-full mx-auto px-6 py-8">
        <Outlet />
      </main>

      <footer className="border-t border-border text-muted text-xs">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <span>TradeX + MEVerse research dashboard</span>
          <span>API: <code className="text-text">/api</code></span>
        </div>
      </footer>
    </div>
  );
}
