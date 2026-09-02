"""腾讯云 AGS（Agent Sandbox）客户端：沙箱工具的创建、查询与实例级换题。

【本文件来源说明】本课题（题目四·强化学习）与「课题三·数据合成」共用同一个腾讯云账号
和同一套 AGS 沙箱基础设施，本文件是从 `课题三-数据合成/swe_synth/clients/ags.py`
原样搬运（vendor）过来的一份独立副本 —— 这样做是为了让本项目（题目四）**自包含**：
客户 clone 本仓库后，`driver.py` 不依赖任何仓库外部的兄弟目录路径就能直接跑，
满足"clone 下来即可远程启动使用"的交付要求。两边逻辑如有分叉，以本文件为准
（本课题不回写课题三）。

Agent2 的「自定义镜像起沙箱」依赖本模块：把题目镜像注册成一个「沙箱工具」，
再用 E2B SDK 以该工具名启动沙箱实例。

API 依据（官方文档，版本 2025-09-20）
------------------------------------
CreateSandboxTool（/document/product/1814/124812）关键参数：
    ToolName             必选  工具名（同一 AppId 下唯一）
    ToolType             必选  枚举含 `custom`（自定义镜像）/ `swebench`
    NetworkConfiguration 必选  { NetworkMode: "PUBLIC" }
    RoleArn              可选  自定义镜像拉取所需角色（缺它会报 MissingParameter.RoleArn）
    CustomConfiguration  可选  { Image, ImageRegistryType, Ports, Resources, Probe, ... }
                              · Image 是镜像地址（不是 ImageUri）
                              · ImageRegistryType 枚举：enterprise / personal / custom
                              · Ports 是数组，元素含 Port(Integer) / Protocol

StartSandboxInstance 同样接受可选的 `CustomConfiguration`，且其中的 `Image`
可以覆盖工具创建时的默认镜像——也就是说切换题目不需要重新创建/删除工具，
只需复用同一个 ToolId、在每次启动实例时传不同的 `Image`（课题三已实测验证）。
这是本课题 `driver.py` 的用法：一个工具、多道题循环，用 `start_instance()`
的 `image_override` 逐题切换。

安全
----
· 凭证（SecretId/Key）只从环境读取，不落盘、不出现在日志
· 所有字段值经 SDK 的 models 对象传递（参数化，无拼接注入面）
"""

from __future__ import annotations

import os
from typing import Any

__all__ = ["AGSClient", "AGSError", "client_of", "models_of"]


class AGSError(RuntimeError):
    """AGS API 调用失败。"""


def client_of(module_name: str, client_cls: str, version: str, region: str):
    """构造某个腾讯云产品的 SDK client（参数列表，无 shell 拼接）。"""
    from tencentcloud.common import credential
    from tencentcloud.common.profile.client_profile import ClientProfile
    from tencentcloud.common.profile.http_profile import HttpProfile

    sid = os.environ.get("TENCENTCLOUD_SECRET_ID", "")
    skey = os.environ.get("TENCENTCLOUD_SECRET_KEY", "")
    if not sid or not skey:
        raise AGSError("未配置 TENCENTCLOUD_SECRET_ID / TENCENTCLOUD_SECRET_KEY（见 .env）")
    cred = credential.Credential(sid, skey)
    hp = HttpProfile(reqTimeout=60)
    cp = ClientProfile(httpProfile=hp)

    mod = __import__(f"tencentcloud.{module_name}.{version}.{module_name}_client",
                     fromlist=[client_cls])
    cls = getattr(mod, client_cls)
    return cls(cred, region, cp)


def models_of(module_name: str, version: str):
    return __import__(f"tencentcloud.{module_name}.{version}.models", fromlist=["models"])


class AGSClient:
    """AGS 沙箱工具管理（创建 / 列表查询）。"""

    VERSION = "v20250920"
    PRODUCT = "ags"

    def __init__(self, region: str | None = None) -> None:
        self.region = region or os.environ.get("TENCENTCLOUD_REGION", "ap-shanghai")
        self._cli = client_of(self.PRODUCT, "AgsClient", self.VERSION, self.region)
        self._m = models_of(self.PRODUCT, self.VERSION)

    # ------------------------------------------------------------ 创建
    def create_tool(
        self,
        tool_name: str,
        image: str,
        *,
        role_arn: str | None = None,
        image_registry_type: str = "personal",   # CCR 个人版
        tool_type: str = "custom",
        description: str = "",
        network_mode: str = "PUBLIC",
        default_timeout: str = "15m",
        command: list[str] | None = None,
        args: list[str] | None = None,
        cpu: str = "2",
        memory: str = "4Gi",
        storage: str | None = None,
        probe_path: str = "/health",
        probe_port: int = 49983,
        probe_ready_timeout_ms: int = 30000,
        probe_timeout_ms: int = 5000,
        probe_period_ms: int = 10000,
        probe_failure_threshold: int = 3,
        probe_success_threshold: int = 1,
        storage_mounts: list[dict[str, Any]] | None = None,
    ) -> str:
        """把镜像注册为沙箱工具，返回 ToolId。自定义镜像必须传 `role_arn`。"""
        role_arn = role_arn or os.environ.get("AGS_ROLE_ARN", "")
        if not role_arn:
            raise AGSError("未配置 AGS_ROLE_ARN（自定义镜像拉取必需，见 .env）")

        m = self._m
        req = m.CreateSandboxToolRequest()
        req.ToolName = tool_name
        req.ToolType = tool_type
        req.Description = description
        req.DefaultTimeout = default_timeout
        req.RoleArn = role_arn

        net = m.NetworkConfiguration()
        net.NetworkMode = network_mode
        req.NetworkConfiguration = net

        if storage_mounts:
            req.StorageMounts = self._build_storage_mounts(storage_mounts)

        req.CustomConfiguration = self._build_custom_configuration(
            image,
            image_registry_type=image_registry_type,
            command=command,
            args=args,
            cpu=cpu,
            memory=memory,
            storage=storage,
            probe_path=probe_path,
            probe_port=probe_port,
            probe_ready_timeout_ms=probe_ready_timeout_ms,
            probe_timeout_ms=probe_timeout_ms,
            probe_period_ms=probe_period_ms,
            probe_failure_threshold=probe_failure_threshold,
            probe_success_threshold=probe_success_threshold,
        )

        rsp = self._cli.CreateSandboxTool(req)
        return getattr(rsp, "ToolId", "")

    # ------------------------------------------------------------ 挂载卷
    def _build_storage_mounts(self, mounts: list[dict[str, Any]]) -> list[Any]:
        m = self._m
        out = []
        for spec in mounts:
            img_src = m.ImageStorageSource()
            img_src.Reference = spec["image"]
            img_src.ImageRegistryType = spec.get("image_registry_type", "personal")
            if spec.get("sub_path"):
                img_src.SubPath = spec["sub_path"]

            src = m.StorageSource()
            src.Image = img_src

            mount = m.StorageMount()
            mount.Name = spec["name"]
            mount.StorageSource = src
            mount.MountPath = spec["mount_path"]
            mount.ReadOnly = spec.get("read_only", True)
            out.append(mount)
        return out

    def _build_mount_options(self, options: list[dict[str, Any]]) -> list[Any]:
        m = self._m
        out = []
        for spec in options:
            opt = m.MountOption()
            opt.Name = spec["name"]
            if spec.get("mount_path"):
                opt.MountPath = spec["mount_path"]
            if spec.get("sub_path"):
                opt.SubPath = spec["sub_path"]
            if "read_only" in spec:
                opt.ReadOnly = spec["read_only"]
            out.append(opt)
        return out

    # ------------------------------------------------------------ 实例级换题（双镜像方案）
    def _build_custom_configuration(
        self,
        image: str,
        *,
        image_registry_type: str = "personal",
        command: list[str] | None = None,
        args: list[str] | None = None,
        cpu: str = "2",
        memory: str = "4Gi",
        storage: str | None = None,
        probe_path: str = "/health",
        probe_port: int = 49983,
        probe_ready_timeout_ms: int = 30000,
        probe_timeout_ms: int = 5000,
        probe_period_ms: int = 10000,
        probe_failure_threshold: int = 3,
        probe_success_threshold: int = 1,
    ):
        m = self._m
        custom = m.CustomConfiguration()
        custom.Image = image
        custom.ImageRegistryType = image_registry_type
        custom.Command = command if command is not None else ["/init"]
        custom.Args = args if args is not None else ["sleep", "infinity"]

        res = m.ResourceConfiguration()
        res.CPU = cpu
        res.Memory = memory
        if storage:
            res.Storage = storage
        custom.Resources = res

        http_get = m.HttpGetAction()
        http_get.Path = probe_path
        http_get.Port = probe_port
        http_get.Scheme = "HTTP"
        probe = m.ProbeConfiguration()
        probe.HttpGet = http_get
        probe.ReadyTimeoutMs = probe_ready_timeout_ms
        probe.ProbeTimeoutMs = probe_timeout_ms
        probe.ProbePeriodMs = probe_period_ms
        probe.FailureThreshold = probe_failure_threshold
        probe.SuccessThreshold = probe_success_threshold
        custom.Probe = probe
        return custom

    def start_instance(
        self,
        tool_id: str,
        *,
        image_override: str | None = None,
        timeout: str = "15m",
        image_registry_type: str = "personal",
        cpu: str = "2",
        memory: str = "4Gi",
        storage: str | None = None,
        mount_options: list[dict[str, Any]] | None = None,
        max_wait: float = 600,
        poll_interval: float = 20,
    ) -> tuple[str, str | None]:
        """启动一个沙箱实例，可选按实例覆盖题目镜像 / 挂载路径。返回
        `(instance_id, effective_image)`。对已知的瞬时性平台错误（镜像预热中 /
        瞬时超时）做限时轮询重试，其余错误直接抛出。"""
        import time

        from tencentcloud.common.exception.tencent_cloud_sdk_exception import (
            TencentCloudSDKException,
        )

        m = self._m
        req = m.StartSandboxInstanceRequest()
        req.ToolId = tool_id
        req.Timeout = timeout
        if image_override:
            req.CustomConfiguration = self._build_custom_configuration(
                image_override,
                image_registry_type=image_registry_type,
                cpu=cpu,
                memory=memory,
                storage=storage,
            )
        if mount_options:
            req.MountOptions = self._build_mount_options(mount_options)

        deadline = time.time() + max_wait
        attempt = 0
        while True:
            attempt += 1
            try:
                rsp = self._cli.StartSandboxInstance(req)
                break
            except TencentCloudSDKException as e:
                code = getattr(e, "code", "") or ""
                message = str(e)
                is_transient = (
                    (code == "ResourceUnavailable" and "still preparing" in message)
                    or (
                        code == "FailedOperation.Timeout"
                        and "Sandbox creation timed out" in message
                    )
                    or "retry later" in message.lower()
                )
                if not is_transient or time.time() >= deadline:
                    raise AGSError(
                        f"StartSandboxInstance 失败（tool_id={tool_id}, "
                        f"image={image_override or '<默认>'}，第 {attempt} 次尝试，"
                        f"code={code}）：{e}"
                    ) from e
                time.sleep(poll_interval)
        inst = rsp.Instance
        effective_image = getattr(
            getattr(inst, "CustomConfiguration", None), "Image", None
        )
        return getattr(inst, "InstanceId", ""), effective_image

    def stop_instance(self, instance_id: str) -> None:
        """停止/回收一个沙箱实例（按实例计费的资源需要显式回收）。"""
        m = self._m
        req = m.StopSandboxInstanceRequest()
        req.InstanceId = instance_id
        self._cli.StopSandboxInstance(req)

    def renew_instance(self, instance_id: str, timeout: str = "24h") -> None:
        """给一个正在运行的实例续期（`UpdateSandboxInstance`）。"""
        m = self._m
        req = m.UpdateSandboxInstanceRequest()
        req.InstanceId = instance_id
        req.Timeout = timeout
        self._cli.UpdateSandboxInstance(req)

    # ------------------------------------------------------------ 查询
    def list_tools(self) -> list[dict[str, Any]]:
        """列出已有沙箱工具（名称 / ID / 类型 / 状态 / 镜像地址）。"""
        m = self._m
        req = m.DescribeSandboxToolListRequest()
        req.Offset, req.Limit = 0, 100
        rsp = self._cli.DescribeSandboxToolList(req)
        out = []
        for t in (rsp.SandboxToolSet or []):
            cc = getattr(t, "CustomConfiguration", None)
            out.append({
                "name": getattr(t, "ToolName", "?"),
                "tool_id": getattr(t, "ToolId", "?"),
                "type": getattr(t, "ToolType", "?"),
                "status": getattr(t, "Status", "?"),
                "image": getattr(cc, "Image", None) if cc else None,
            })
        return out

    def find_tool(self, tool_name: str) -> dict[str, Any] | None:
        for t in self.list_tools():
            if t["name"] == tool_name:
                return t
        return None

    # ------------------------------------------------------------ 删除
    def delete_tool(self, tool_id: str) -> None:
        """删除沙箱工具（镜像有更新后必须删除重建，CCR tag 内容不会自动刷新）。"""
        m = self._m
        req = m.DeleteSandboxToolRequest()
        req.ToolId = tool_id
        self._cli.DeleteSandboxTool(req)

    # ------------------------------------------------------------ 等待就绪
    def wait_tool_active(
        self, tool_name: str, *, timeout: float = 180, interval: float = 3,
    ) -> dict[str, Any]:
        """轮询直到工具状态变为 ``ACTIVE``（`CreateSandboxTool` 是异步的）。"""
        import time

        deadline = time.time() + timeout
        last: dict[str, Any] | None = None
        while time.time() < deadline:
            last = self.find_tool(tool_name)
            if last is None:
                raise AGSError(f"工具 {tool_name} 未找到（可能创建失败或已被删除）")
            status = str(last.get("status", "")).upper()
            if status == "ACTIVE":
                return last
            if status in ("FAILED", "ERROR", "DELETED", "DELETING"):
                raise AGSError(f"工具 {tool_name} 未能就绪，状态：{status}")
            time.sleep(interval)
        raise AGSError(
            f"等待工具 {tool_name} 变为 ACTIVE 超时（{timeout}s），"
            f"当前状态：{last.get('status') if last else '未知'}"
        )
