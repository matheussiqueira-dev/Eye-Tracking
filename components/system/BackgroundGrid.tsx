/** Fixed full-viewport decorative grid background with drift animation. */
import styles from "./BackgroundGrid.module.css";

export function BackgroundGrid() {
  return <div className={styles.grid} aria-hidden="true" />;
}
