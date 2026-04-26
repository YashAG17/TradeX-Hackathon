import { NavLink } from "react-router-dom";

export interface TabItem {
  to: string;
  label: string;
}

interface TabsProps {
  items: TabItem[];
}

const tabClass = ({ isActive }: { isActive: boolean }) =>
  [
    "px-4 py-2 -mb-px border-b-2 text-sm font-medium transition-colors",
    isActive
      ? "border-accent text-accent"
      : "border-transparent text-muted hover:text-text",
  ].join(" ");

export function Tabs({ items }: TabsProps) {
  return (
    <div className="border-b border-border mb-6 flex gap-2 overflow-x-auto">
      {items.map((item) => (
        <NavLink key={item.to} to={item.to} className={tabClass} end>
          {item.label}
        </NavLink>
      ))}
    </div>
  );
}
