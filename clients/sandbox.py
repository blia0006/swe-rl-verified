"""沙箱实例启动的统一封装：处理 AGS 的镜像预热等待。

## 为什么需要这一层（实测踩坑）

AGS 首次使用某个新推送的镜像时，需要在平台侧做一次「镜像预热」，期间：

    第 1~2 次调用 → `ResourceUnavailable: image is still preparing, please retry later`
    更早的时候   → `FailedOperation.ImagePrepare: Internal server error`

⚠️ **后者极具误导性** —— 它长得像故障，实际只是预热未完成。
本轮曾据此误判为"手工构造的镜像清单不被平台接受"，并因此改写了两版
镜像融合方案（正向 OCI 加层 → docker build），走了不少弯路。
真相是：**镜像本身一直没问题，只需要等约 4 分钟**（实测第 3 次尝试成功，
每次间隔 ~80s）。

因此把"预热等待"固化在这里，避免任何调用方再把它当成错误。

## 第二个坑：新实例的 DNS 传播延迟

`Sandbox.connect(instance_id)` 走的是 `<instance>-49999.<region>.tencentags.com`
这个**按实例动态生成的域名**。实例刚创建时，该域名尚未在公网 DNS 生效，报

    ConnectError: [Errno 8] nodename nor servname provided, or not known

它看起来像"沙箱起失败了"，实际实例已 RUNNING，只是域名还没传播开。
实测同一份代码连续两次运行、一次成功一次失败，正是这个原因。
`connect_with_retry` 专门吸收这段延迟。
"""

from __future__ import annotations

import time
from typing import Any

# 这两类错误都表示「镜像还在预热」，应当继续等待而非放弃
_WARMUP_MARKERS = (
    "still preparing",
    "ImagePrepare",
    "image is preparing",
)

# 这些错误表示「实例域名尚未在 DNS 生效」，等待后重试即可
_DNS_MARKERS = (
    "nodename nor servname",
    "Name or service not known",
    "Temporary failure in name resolution",
    "getaddrinfo",
    "ConnectError",
)


def is_dns_pending(err: BaseException) -> bool:
    msg = str(err)
    return any(m in msg for m in _DNS_MARKERS)


def connect_with_retry(
    instance_id: str,
    *,
    max_wait_s: float = 120,
    poll_s: float = 5,
    verbose: bool = False,
) -> Any:
    """连接沙箱实例，吸收 DNS 传播延迟。

    只对 DNS 类错误重试；其他错误（如鉴权失败、实例不存在）立即抛出，
    避免把真故障拖成 2 分钟超时。
    """
    from e2b_code_interpreter import Sandbox

    deadline = time.time() + max_wait_s
    attempt = 0
    last: BaseException | None = None
    while time.time() < deadline:
        attempt += 1
        try:
            return Sandbox.connect(instance_id)
        except Exception as e:  # noqa: BLE001 —— 需按错误内容分流
            last = e
            if not is_dns_pending(e):
                raise
            if verbose:
                print(
                    f"      实例域名 DNS 未生效（第 {attempt} 次），"
                    f"剩余 {int(deadline - time.time())}s",
                    flush=True,
                )
            time.sleep(poll_s)
    assert last is not None
    raise TimeoutError(f"实例 {instance_id} 域名 {max_wait_s}s 内未生效：{last}")


def is_warming_up(err: BaseException) -> bool:
    msg = str(err)
    return any(m in msg for m in _WARMUP_MARKERS)


def start_instance_with_warmup(
    ags: Any,
    tool_id: str,
    image: str,
    *,
    max_wait_s: float = 900,
    poll_s: float = 20,
    per_try_wait: float = 60,
    verbose: bool = True,
    **kwargs: Any,
) -> tuple[str, str | None]:
    """启动沙箱实例，自动等待镜像预热完成。

    Args:
        ags: `clients.ags.AGSClient` 实例
        tool_id: 沙箱工具 ID（全程复用同一个，靠 image_override 切题）
        image: 题目镜像地址
        max_wait_s: 总等待上限。新镜像首次预热实测约 4 分钟，留足余量
        poll_s: 两次尝试之间的间隔
        per_try_wait: 单次 start_instance 内部的等待上限（调小可快速拿到预热状态）

    Returns:
        (instance_id, effective_image)

    Raises:
        最后一次的原始异常（非预热类错误会立即抛出，不做无谓重试）
    """
    deadline = time.time() + max_wait_s
    attempt = 0
    last_exc: BaseException | None = None

    while time.time() < deadline:
        attempt += 1
        try:
            return ags.start_instance(
                tool_id, image_override=image, max_wait=per_try_wait, **kwargs
            )
        except Exception as e:  # noqa: BLE001 — 需按错误内容分流
            last_exc = e
            if not is_warming_up(e):
                raise  # 真错误：立即抛出，不浪费时间重试
            if verbose:
                remain = int(deadline - time.time())
                print(
                    f"      镜像预热中（第 {attempt} 次），剩余等待预算 {remain}s",
                    flush=True,
                )
            time.sleep(poll_s)

    assert last_exc is not None
    raise TimeoutError(
        f"镜像预热超过 {max_wait_s}s 仍未就绪：{image}\n最后一次错误：{last_exc}"
    )
