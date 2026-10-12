import { cn } from "@/utils/cn";

export function BrandMark({ small = false }: { small?: boolean }) {
  return (
    <span
      className={cn("brand-mark", small && "brand-mark-small")}
      aria-hidden="true"
    >
      <span />
      <span />
      <span />
    </span>
  );
}
