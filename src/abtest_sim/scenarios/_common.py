from ..runner import Block, Task

BOOT_CHUNK = 50  # бутстрап медленный, мелкие чанки равномернее грузят процессы


def boot_task(name, metric, n, uplift, param, n_reps, **kwargs):
    """Задача для бутстрапа: своя метка в param и маленькие чанки."""
    kwargs["boot"] = True
    return Task(name, metric, n, uplift, param, n_reps, chunk_size=BOOT_CHUNK, kwargs=kwargs)


def to_block(test, res, truth=None):
    """Результат критерия в Block. Если известна истинная разница,
    отмечаем, попала ли она в доверительный интервал."""
    covers = None
    if truth is not None:
        covers = (res.ci_low <= truth) & (truth <= res.ci_high)
    return Block(test, res.p_value, effect=res.effect, ci_covers=covers)
