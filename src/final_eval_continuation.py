"""Deterministic reserved-period continuation for the frozen V2 simulator.

This module is predeclared before final-period outcomes are exposed. It leaves
all original development draws untouched, then appends new random draws only
after the frozen development RNG sequence has been consumed.
"""

import sys
import types


def load_module(name, source):
    module = types.ModuleType(name)
    module.__file__ = f"<{name}>"
    # dataclasses inspects sys.modules[cls.__module__] while decorating classes.
    # Register the transient module before exec so the frozen Config dataclass can
    # be created correctly on Python 3.11-3.13. This changes no experiment logic.
    sys.modules[name] = module
    try:
        exec(compile(source, module.__file__, "exec"), module.__dict__)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f"Frozen-engine marker changed: {old!r}")
    return source.replace(old, new, 1)


def make_final_engine_source(frozen_source):
    source = frozen_source
    source = replace_once(
        source,
        '        travel_country = str(rng.choice([c for c in COUNTRIES if c != p["home"]]))\n        profiles.append(p)\n',
        '        travel_country = str(rng.choice([c for c in COUNTRIES if c != p["home"]]))\n'
        '        p["_dev_travel_day"] = travel_day\n'
        '        profiles.append(p)\n',
    )
    source = replace_once(
        source,
        '        if rng.random() < 0.35:\n            day = int(rng.integers(10, dev_days))\n',
        '        p["_had_dev_burst"] = bool(rng.random() < 0.35)\n'
        '        if p["_had_dev_burst"]:\n'
        '            day = int(rng.integers(10, dev_days))\n',
    )

    extension = r'''
    # FINAL_EVALUATION_CONTINUATION_V1
    # The frozen simulator above has fully consumed its development RNG stream.
    # Only now is the horizon opened and additional reserved-period draws made.
    horizon = cfg.days * 86400
    final_days = cfg.days - dev_days
    extension_share = final_days / dev_days
    phone_probability = 1 - (1 - 0.30) ** extension_share
    travel_probability = 1 - (1 - 0.25) ** extension_share
    burst_probability = 1 - (1 - 0.35) ** extension_share
    attack_probability = 1 - (1 - cfg.fraud_customer_fraction) ** extension_share

    for i, p in enumerate(profiles):
        prior_phone_change = p["phone_day"] < dev_days
        final_phone_day = None
        if not prior_phone_change and rng.random() < phone_probability:
            final_phone_day = int(rng.integers(dev_days, cfg.days))

        prior_travel = p.get("_dev_travel_day", dev_days + 1) < dev_days
        final_travel_day = None
        final_travel_country = None
        if not prior_travel and rng.random() < travel_probability:
            final_travel_day = int(rng.integers(dev_days, cfg.days))
            final_travel_country = str(rng.choice([c for c in COUNTRIES if c != p["home"]]))

        n = int(rng.poisson(p["rate"] * final_days))
        days = rng.integers(dev_days, cfg.days, n)
        hours = np.mod(rng.normal(p["hour"], 3, n), 24)
        for day, hour in zip(days, hours):
            context, country = "routine", p["home"]
            changed = prior_phone_change or (final_phone_day is not None and day >= final_phone_day)
            device = p["device1"] if changed else p["device0"]
            if final_phone_day is not None and final_phone_day <= day < final_phone_day + 5:
                context = "new_phone"
            if final_travel_day is not None and final_travel_day <= day < final_travel_day + 6:
                context, country = "travel", final_travel_country
            if rng.random() < 0.025:
                device = "household_%04d" % (i // 4)
            amount = p["base"] * rng.lognormal(0, p["spread"])
            if rng.random() < 0.018:
                amount *= rng.uniform(2, 7)
                context = "large_purchase"
            merchant = rng.choice(p["favourites"] if rng.random() < 0.85 else merchant_ids)
            add(p, day * 86400 + int(hour * 3600), amount, device, country,
                merchant, 0, "legitimate", context, 0.05, rng)

        if not p.get("_had_dev_burst", False) and rng.random() < burst_probability:
            day = int(rng.integers(dev_days, cfg.days))
            changed = prior_phone_change or (final_phone_day is not None and day >= final_phone_day)
            device = p["device1"] if changed else p["device0"]
            for j in range(int(rng.integers(4, 11))):
                add(p, day * 86400 + p["hour"] * 3600 + j * 70,
                    p["base"] * rng.uniform(0.08, 2.2), device, p["home"],
                    rng.choice(p["favourites"]), 0, "legitimate", "legitimate_burst", 0.25, rng)

    selected_dev = {int(x) for x in selected}
    eligible = np.array([i for i in range(len(profiles)) if i not in selected_dev], dtype=int)
    final_attack_count = min(len(eligible), max(3, int(round(len(eligible) * attack_probability))))
    selected_final = attack_rng.choice(eligible, final_attack_count, replace=False)

    for k, index in enumerate(selected_final):
        p = profiles[int(index)]
        scenario = ("account_takeover", "card_testing", "low_and_slow")[k % 3]
        day = int(attack_rng.integers(dev_days, cfg.days))
        mimic = hardened and attack_rng.random() < 0.80
        hour = float(np.mod(attack_rng.normal(p["hour"], 3), 24)) if mimic else float(attack_rng.uniform(0, 24))
        base_sec = day * 86400 + int(hour * 3600)
        previous = [r for r in normal_histories[p["customer_id"]] if r["second"] < base_sec]
        latest = max(previous, key=lambda r: r["second"]) if previous else None
        familiar_device = latest["device_id"] if latest else p["device0"]
        device = familiar_device if attack_rng.random() < (0.85 if hardened else 0.45) else "shared_actor_%03d" % int(attack_rng.integers(0, 30))
        country = p["home"] if attack_rng.random() < (0.92 if hardened else 0.75) else str(attack_rng.choice(COUNTRIES))
        known_merchants = [r["merchant_id"] for r in previous] or list(p["favourites"])

        if scenario == "account_takeover":
            count, step = int(attack_rng.integers(2, 7)), int(attack_rng.integers(120, 1800))
            if mimic:
                step = int(attack_rng.integers(1800, 36000))
            fail_p = 0.05 if mimic else 0.18
        elif scenario == "card_testing":
            count, step = int(attack_rng.integers(5, 13)), int(attack_rng.integers(15, 150))
            if hardened and attack_rng.random() < 0.50:
                step = int(attack_rng.integers(600, 5400))
            fail_p = float(attack_rng.uniform(0.02, 0.12)) if mimic else 0.45
        else:
            count, step, fail_p = int(attack_rng.integers(3, 9)), int(attack_rng.integers(1, 4)) * 86400, 0.06

        for j in range(count):
            event_sec = base_sec + j * step
            if hardened and scenario == "low_and_slow":
                event_sec = max(base_sec, event_sec + int(attack_rng.normal(0, 4 * 3600)))
            if mimic and scenario != "card_testing":
                amount = p["base"] * attack_rng.lognormal(0, p["spread"])
            elif scenario == "account_takeover":
                amount = p["base"] * attack_rng.uniform(1.2, 6)
            elif scenario == "card_testing":
                amount = p["base"] * (attack_rng.lognormal(-0.2, p["spread"]) if mimic and attack_rng.random() < 0.55 else attack_rng.uniform(0.02, 0.35))
            else:
                amount = p["base"] * attack_rng.lognormal(0, p["spread"] * 0.7)
            merchant = attack_rng.choice(known_merchants if attack_rng.random() < (0.80 if hardened else 0.35) else merchant_ids)
            add(p, event_sec, amount, device, country, merchant, 1, scenario, "not_applicable", fail_p, attack_rng)
'''

    source = replace_once(source, "    tx = pd.DataFrame(events)\n", extension + "\n    tx = pd.DataFrame(events)\n")
    return source
