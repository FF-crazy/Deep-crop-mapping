
"""
DeepCropMapping TUI
交互式文本界面，用于农作物遥感数据分析和深度学习模型训练

Author: DeepCropMapping Project
"""

import sys
from pathlib import Path
from typing import ClassVar

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from textual.app import App, ComposeResult
from textual.containers import Container, VerticalScroll
from textual.widgets import (
    Header, Footer, Button, DataTable, Static,
    Label, Placeholder, Log, Markdown, Tabs, Tab,
    ProgressBar, Input
)
from textual.reactive import reactive
from textual.binding import Binding

from rich.panel import Panel
from rich.text import Text
from rich.align import Align

# 为了避免在没有显示环境的服务器上运行出错，先进行配置
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# --- 数据处理模块 ---
try:
    from deepcropmapping.data_visualization import CropDataVisualizer
except ImportError:
    print("错误：无法导入data_visualization模块。")
    print("请确保'deepcropmapping/data_visualization.py'文件存在且无误。")
    sys.exit(1)


class WelcomeScreen(Static):
    """欢迎界面"""
    def compose(self) -> ComposeResult:
        banner_text = """
        ██████╗ ███████╗███████╗██████╗  ██████╗ ██████╗ ██████╗ ██╗   ██╗██████╗ ██████╗ ██╗██╗   ██╗███╗   ██╗ ██████╗
        ██╔══██╗██╔════╝██╔════╝██╔══██╗██╔════╝██╔═══██╗██╔══██╗██║   ██║██╔══██╗██╔══██╗██║██║   ██║████╗  ██║██╔════╝
        ██║  ██║█████╗  █████╗  ██████╔╝██║     ██║   ██║██████╔╝██║   ██║██████╔╝██████╔╝██║██║   ██║██╔██╗ ██║██║  ███╗
        ██║  ██║██╔══╝  ██╔══╝  ██╔═══╝ ██║     ██║   ██║██╔══██╗██║   ██║██╔═══╝ ██╔═══╝ ██║██║   ██║██║╚██╗██║██║   ██║
        ██████╔╝███████╗███████╗██║     ╚██████╗╚██████╔╝██║  ██║╚██████╔╝██║     ██║     ██║╚██████╔╝██║ ╚████║╚██████╔╝
        ╚═════╝ ╚══════╝╚══════╝╚═╝      ╚═════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝
        """
        yield Static(banner_text, classes="banner")
        yield Static("Version 1.0.0 | Press any key to start", classes="version-info")
        

class DataAnalysisScreen(Static):
    """数据分析界面"""
    
    # 类别颜色定义
    COLORS = ["#FFD700", "#8B4513", "#FFA500", "#FF6347", "#808080", "#0000FF", "#696969", "#9ACD32"]
    
    def __init__(self, data_dir: str = "./dataset", **kwargs):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.visualizer = None
        self.log_widget = None

    def on_mount(self) -> None:
        """组件挂载后执行"""
        self.log_widget = self.query_one(Log)
        
        try:
            self.log_widget.write(f"正在从'{self.data_dir}'加载数据集...")
            self.visualizer = CropDataVisualizer(data_dir=self.data_dir)
            self.log_widget.write("数据集加载成功！")
            self.populate_overview_tab()
        except FileNotFoundError:
            self.log_widget.write(f"[bold red]错误: 数据集未在'{self.data_dir}'中找到。[/bold red]")
        except Exception as e:
            self.log_widget.write(f"[bold red]加载数据集时发生错误: {e}[/bold red]")

    def populate_overview_tab(self) -> None:
        """填充数据概览标签页"""
        if not self.visualizer:
            return
            
        # 填充表格
        table = self.query_one("#overview_table", DataTable)
        table.add_columns("属性 (Property)", "值 (Value)")
        
        # X数据
        x = self.visualizer.x_data
        table.add_row("特征数据 (X) 维度", str(x.shape))
        table.add_row("数据类型", str(x.dtype))
        table.add_row("内存占用", f"{x.nbytes / 1024**2:.2f} MB")
        table.add_row("值域 (Min, Max)", f"[{x.min()}, {x.max()}]")
        
        # Y数据
        y = self.visualizer.y_data
        table.add_row("标签数据 (Y) 维度", str(y.shape))
        table.add_row("数据类型", str(y.dtype))
        table.add_row("唯一标签", str(np.unique(y)))

        # 类别分布
        unique_labels, counts = np.unique(y, return_counts=True)
        dist_table = self.query_one("#distribution_table", DataTable)
        dist_table.add_columns("类别 (Class)", "像素数 (Pixels)", "百分比 (Percentage)")
        
        for label, count in zip(unique_labels, counts):
            name = self.visualizer.crop_labels.get(label, "未知")
            percentage = f"{count * 100 / y.size:.2f}%"
            # 使用Rich Text来为类别名称添加颜色
            colored_name = f"[{self.COLORS[label % len(self.COLORS)]}]{name}[/]"
            dist_table.add_row(colored_name, f"{count:,}", percentage)
            
        self.log_widget.write("数据概览已更新。")

    def compose(self) -> ComposeResult:
        yield Header(name="DeepCropMapping - 数据分析")
        with Container():
            with Tabs(id="data_tabs"):
                with Tab("数据概览 (Overview)", id="tab_overview"):
                    with VerticalScroll(classes="main-container"):
                        yield Label("数据集基本信息", classes="title")
                        yield DataTable(id="overview_table")
                        yield Label("类别分布", classes="title")
                        yield DataTable(id="distribution_table")

                with Tab("空间分布 (Spatial)", id="tab_spatial"):
                    yield Label("空间分布图 (功能待实现)", classes="title")
                    yield Placeholder("这里将显示空间分布图")
                
                with Tab("光谱分析 (Spectral)", id="tab_spectral"):
                    yield Label("光谱分析 (功能待实现)", classes="title")
                    yield Placeholder("这里将显示光谱分析图")
                    
                with Tab("时序分析 (Temporal)", id="tab_temporal"):
                    yield Label("时序分析 (功能待实现)", classes="title")
                    yield Placeholder("这里将显示时序分析图")
            
            yield Log(id="data_log", classes="log-panel", auto_scroll=True, name="日志")
        yield Footer()

class MainApp(App):
    """DeepCropMapping主应用"""

    TITLE = "DeepCropMapping"
    SUB_TITLE = "交互式深度学习农作物分类工具"
    
    CSS_PATH = "tui_style.css"
    
    BINDINGS = [
        Binding("d", "toggle_dark", "切换深/浅色模式"),
        Binding("q", "quit", "退出应用", priority=True)
    ]
    
    def compose(self) -> ComposeResult:
        # 初始界面为欢迎屏幕
        yield WelcomeScreen(id="welcome_screen")

    def on_key(self) -> None:
        """任意按键后切换到主界面"""
        welcome = self.query_one("#welcome_screen", WelcomeScreen)
        if welcome.display:
            welcome.display = False
            # 挂载主界面
            self.mount(Header(name=self.TITLE))
            self.mount(Container(
                DataAnalysisScreen(id="data_analysis_screen"),
                # 未来可以添加其他屏幕
                # Placeholder("模型训练", id="train_screen", classes="hidden"),
                # Placeholder("模型评估", id="eval_screen", classes="hidden")
            ))
            self.mount(Footer())
    
    def action_toggle_dark(self) -> None:
        """切换深/浅色模式的动作"""
        self.theme = "textual-dark" if self.theme == "textual-light" else "textual-light"

def main():
    """主函数入口"""
    app = MainApp()
    app.run()

if __name__ == "__main__":
    main()
''