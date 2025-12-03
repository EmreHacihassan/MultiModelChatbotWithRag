import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    open: true,
    proxy: {
      "/models": "http://localhost:8000",
      "/sessions": "http://localhost:8000",
      "/chat": "http://localhost:8000",
      "/chat/stream": "http://localhost:8000",
      "/api": "http://localhost:8000",
      "/ws": {
        target: "http://localhost:8000",
        ws: true,
        changeOrigin: true,
      },
    },
  },
});