import styles from "./Footer.module.css";

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
        >
          https://www.matheussiqueira.dev/
        </a>
      </div>
    </footer>
  );
}
