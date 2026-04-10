type Falsy = false | null | undefined;

/**
 * Concatenate CSS class names, filtering out falsy values.
 *
 * @param names - Class name segments (may include booleans for conditional classes).
 * @returns Space-joined string of truthy class names.
 *
 * @example
 * cx(styles.base, isActive && styles.active)
 */
export function cx(...names: Array<string | Falsy>): string {
  return names.filter(Boolean).join(" ");
}
