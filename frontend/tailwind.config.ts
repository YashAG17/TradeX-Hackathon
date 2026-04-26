import type { Config } from "tailwindcss";

// Mirrors the COLORS palette in dashboard.py:38 so server-rendered Plotly figures
// and the React UI use the same hex codes.
const config: Config = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#0f1117",
        surface: "#1a1d27",
        border: "#2a2d3a",
        muted: "#8b8fa3",
        text: "#e4e6eb",
        accent: {
          DEFAULT: "#00c9a7",
          hover: "#00b396",
        },
        accent2: "#0088cc",
        danger: "#ff4757",
        warning: "#ffa502",
        success: "#2ed573",
        info: "#3498db",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
      },
      keyframes: {
        chartFadeUp: {
          from: { opacity: "0", transform: "translateY(24px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
        chartScaleIn: {
          from: { opacity: "0", transform: "scale(0.92)" },
          to: { opacity: "1", transform: "scale(1)" },
        },
        gaugePopIn: {
          "0%": { opacity: "0", transform: "scale(0.6)" },
          "60%": { opacity: "1", transform: "scale(1.04)" },
          "100%": { opacity: "1", transform: "scale(1)" },
        },
        tableSlideIn: {
          from: { opacity: "0", transform: "translateY(16px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        "fade-up": "chartFadeUp 0.7s ease-out both",
        "scale-in": "chartScaleIn 0.6s ease-out both",
        "gauge-pop": "gaugePopIn 0.8s cubic-bezier(0.34, 1.56, 0.64, 1) both",
        "table-slide": "tableSlideIn 0.5s ease-out both",
      },
    },
  },
  plugins: [],
};

export default config;
