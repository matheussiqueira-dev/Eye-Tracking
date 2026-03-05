import styles from "./WhatsAppButton.module.css";

export function WhatsAppButton() {
  return (
    <a
      className={styles.button}
      href="https://wa.me/5581999203683"
      target="_blank"
      rel="noreferrer"
      aria-label="Abrir conversa no WhatsApp"
    >
      <svg aria-hidden="true" viewBox="0 0 64 64">
        <path
          d="M32 12c-11 0-20 9-20 20 0 4 1.2 8 3.5 11.4L12 52l8.8-3.2A20 20 0 1 0 32 12Z"
          fill="none"
          stroke="currentColor"
          strokeWidth="4"
          strokeLinejoin="round"
        />
        <path
          d="M24 24c1 8 8 15 16 16l4-4c1-1 2-1 3-.7l5 2.1c1 .4 1.5 1.5 1.3 2.6-.8 4.7-4.9 8-9.7 7.5C26 46.3 17.7 38 16.5 27.4c-.5-4.8 2.8-8.9 7.5-9.7 1.1-.2 2.2.3 2.6 1.3l2.1 5c.4 1 .2 2-.7 3L24 24Z"
          fill="currentColor"
        />
      </svg>
    </a>
  );
}
