import { Check, ChevronDown, Cpu, X } from "lucide-react";
import { machines } from "@/fixtures/seed";
import type { MachineScope } from "@/types";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuCheckboxItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

export function MachinePicker({
  value,
  onChange,
}: {
  value: MachineScope[];
  onChange(value: MachineScope[]): void;
}) {
  return (
    <div className="machine-picker">
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="outline" className="machine-trigger">
            <Cpu size={16} />
            {value.length === 1
              ? value[0].name
              : value.length > 1
                ? `${value.length} selected`
                : "Not identified"}
            <ChevronDown size={14} className="ml-auto" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start" className="w-72">
          <DropdownMenuLabel>Specific machines</DropdownMenuLabel>
          {machines
            .filter((m) => m.kind === "asset")
            .map((machine) => (
              <DropdownMenuCheckboxItem
                key={machine.id}
                checked={value.some((m) => m.id === machine.id)}
                onSelect={(event) => event.preventDefault()}
                onCheckedChange={(checked) =>
                  onChange(
                    checked
                      ? [...value, machine]
                      : value.filter((m) => m.id !== machine.id),
                  )
                }
              >
                <span>
                  <span className="block">{machine.name}</span>
                  <span className="text-xs text-muted-foreground">
                    {machine.location ?? machine.family}
                  </span>
                </span>
              </DropdownMenuCheckboxItem>
            ))}
          <DropdownMenuSeparator />
          <DropdownMenuLabel>Machine families</DropdownMenuLabel>
          {machines
            .filter((m) => m.kind === "family")
            .map((machine) => (
              <DropdownMenuCheckboxItem
                key={machine.id}
                checked={value.some((m) => m.id === machine.id)}
                onSelect={(event) => event.preventDefault()}
                onCheckedChange={(checked) =>
                  onChange(
                    checked
                      ? [...value, machine]
                      : value.filter((m) => m.id !== machine.id),
                  )
                }
              >
                {machine.name}
              </DropdownMenuCheckboxItem>
            ))}
          <DropdownMenuSeparator />
          <DropdownMenuCheckboxItem
            checked={!value.length}
            onCheckedChange={() => onChange([])}
          >
            Not identified
          </DropdownMenuCheckboxItem>
        </DropdownMenuContent>
      </DropdownMenu>
      {!!value.length && (
        <div className="scope-chips">
          {value.map((machine) => (
            <span key={machine.id} className="scope-chip">
              <Check size={12} />
              {machine.name}
              <button
                type="button"
                aria-label={`Remove ${machine.name}`}
                onClick={() =>
                  onChange(value.filter((m) => m.id !== machine.id))
                }
              >
                <X size={12} />
              </button>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
