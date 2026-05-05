// @ts-check
// Desenvolvido por Matheus Siqueira - www.matheussiqueira.dev
import nextCoreWebVitals from "eslint-config-next/core-web-vitals";
import nextTypescript from "eslint-config-next/typescript";

/** @type {import("eslint").Linter.FlatConfig[]} */
const config = [
  {
    ignores: [".next/**", "coverage/**", "node_modules/**"],
  },
  ...nextCoreWebVitals,
  ...nextTypescript,
];

export default config;
