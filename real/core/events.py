"""Declarative, source-filtered event bindings, matching sonic_deploy's shape."""
from dataclasses import replace


SYSTEM_EVENTS = frozenset({"estop", "shutdown", "reset", "discontinuity"})


class EventBindings:
    def __init__(self, rules, *, sources, actions=None):
        self.rules = []
        if not isinstance(rules, list):
            raise ValueError("events must be a list of {on, sources, action} bindings")
        for rule in rules:
            if not isinstance(rule, dict):
                raise ValueError("Each event binding must be a mapping")
            # PyYAML's YAML 1.1 resolver reads an unquoted `on` key as True.
            rule = dict(rule)
            if True in rule:
                rule["on"] = rule.pop(True)
            if set(rule) - {"on", "sources", "action", "payload"} or not {"on", "action"} <= rule.keys():
                raise ValueError(f"Invalid event binding: {rule}")
            if not all(isinstance(rule[key], str) and rule[key] for key in ("on", "action")):
                raise ValueError("Event names and actions must be nonempty strings")
            selected = rule.get("sources", [])
            if (not isinstance(selected, list) or not all(isinstance(name, str) for name in selected)
                    or set(selected) - sources):
                raise ValueError(f"Unknown event sources: {selected}")
            if actions is not None and rule["action"] not in actions:
                raise ValueError(f"Unknown event action: {rule['action']}")
            self.rules.append(rule)

    def dispatch(self, event):
        for rule in self.rules:
            if event.kind == rule["on"] and (not rule.get("sources") or event.source in rule["sources"]):
                yield replace(event, kind=rule["action"], payload=rule.get("payload", event.payload))
