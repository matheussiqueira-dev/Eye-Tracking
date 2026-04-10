import styles from "./Footer.module.css";

/** Site footer with author attribution and profile link. */
export function Footer() {
  return (
    <footer className={styles.footer}>
      <div className={styles.inner}>
        <span>Desenvolvido por Matheus Siqueira</span>
        <a
          className={styles.link}
          href="https://www.matheussiqueira.dev/"
          target="_blank"
          rel="noreferrer"
          aria-label="Matheus Siqueira portfolio (opens in new tab)"
        >
          https://www.matheussiqueira.dev/
        </a>
      </div>
    </footer>
  );
}
