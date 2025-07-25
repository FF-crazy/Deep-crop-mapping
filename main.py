#!/usr/bin/env python3
"""
DeepCropMapping 简洁TUI
基于命令行风格的交互界面，用于农作物遥感数据分析和深度学习模型训练

Author: DeepCropMapping Project
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Static, Input
from textual.reactive import reactive
from textual.binding import Binding

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


class SimpleTUI(App):
    """DeepCropMapping 简洁TUI界面"""

    TITLE = "DeepCropMapping - 农作物遥感数据分析工具"
    
    # 响应式状态
    current_menu = reactive("main")
    data_loaded = reactive(False)
    
    BINDINGS = [
        Binding("q", "quit", "退出", priority=True),
        Binding("ctrl+c", "quit", "退出", priority=True),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.visualizer = None
        self.dataset_info = {}
        
    def compose(self) -> ComposeResult:
        """构建界面"""
        with Vertical():
            yield Static(self.get_header(), id="header")
            yield Static("", id="separator1")
            yield Static(self.get_info_display(), id="info")
            yield Static("", id="separator2")
            yield Static(self.get_menu_options(), id="menu")
            yield Static("", id="separator3")
            yield Static("请输入选项数字后按回车:", id="prompt")
            yield Input(placeholder="输入数字...", id="command_input")
            yield Static("", id="separator4")
            yield Static("就绪 | q=退出", id="status")

    def get_header(self) -> str:
        """获取头部信息"""
        return """
╔══════════════════════════════════════════════════════════════════╗
║               DeepCropMapping v1.0.0                            ║
║           农作物遥感数据分析 & 深度学习建模工具                    ║
╚══════════════════════════════════════════════════════════════════╝
        """

    def get_info_display(self) -> str:
        """获取信息显示内容"""
        if not self.data_loaded:
            return """
📊 数据集状态: 未加载
📁 数据目录: ./dataset
🔄 请选择 "1" 加载数据集以查看详细信息
            """
        
        # 如果数据已加载，显示数据集信息
        info = "📊 数据集信息:\n"
        for key, value in self.dataset_info.items():
            info += f"   {key}: {value}\n"
        return info

    def get_menu_options(self) -> str:
        """获取菜单选项"""
        if self.current_menu == "main":
            return """
╔══════════════ 主菜单 ══════════════╗
║  1. 加载数据集                     ║
║  2. 数据概览                       ║
║  3. 类别分布分析                   ║
║  4. 空间分布可视化                 ║
║  5. 光谱特征分析                   ║
║  6. 模型训练 (开发中)              ║
║  7. 模型评估 (开发中)              ║
║  8. 设置                          ║
║  q. 退出程序                       ║
╚══════════════════════════════════╝
            """
        elif self.current_menu == "data_overview":
            return """
╔══════════════ 数据概览 ══════════════╗
║  1. 显示基本统计信息                ║
║  2. 显示数据维度信息                ║
║  3. 显示内存使用情况                ║
║  4. 显示数据类型信息                ║
║  0. 返回主菜单                      ║
╚════════════════════════════════════╝
            """
        elif self.current_menu == "class_analysis":
            return """
╔══════════════ 类别分布分析 ══════════════╗
║  1. 显示类别统计表                      ║
║  2. 生成类别分布图                      ║
║  3. 显示类别平衡性分析                  ║
║  0. 返回主菜单                          ║
╚════════════════════════════════════════╝
            """
        else:
            return self.get_menu_options()  # 默认返回主菜单

    def on_mount(self) -> None:
        """应用启动时执行"""
        self.update_status("应用已启动，请选择操作")
        # 确保输入框获得焦点
        input_widget = self.query_one("#command_input", Input)
        input_widget.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """处理用户输入"""
        command = event.value.strip()
        if not command:
            return
            
        # 清空输入框
        self.query_one("#command_input", Input).value = ""
        
        # 处理命令
        self.handle_command(command)

    def handle_command(self, command: str) -> None:
        """处理用户命令"""
        try:
            if self.current_menu == "main":
                self.handle_main_menu(command)
            elif self.current_menu == "data_overview":
                self.handle_data_overview_menu(command)
            elif self.current_menu == "class_analysis":
                self.handle_class_analysis_menu(command)
        except Exception as e:
            self.update_status(f"错误: {str(e)}")

    def handle_main_menu(self, command: str) -> None:
        """处理主菜单命令"""
        if command == "1":
            self.load_dataset()
        elif command == "2":
            if not self.data_loaded:
                self.update_status("请先加载数据集")
                return
            self.current_menu = "data_overview"
            self.refresh_display()
        elif command == "3":
            if not self.data_loaded:
                self.update_status("请先加载数据集")
                return
            self.current_menu = "class_analysis"
            self.refresh_display()
        elif command == "4":
            self.update_status("空间分布可视化功能开发中...")
        elif command == "5":
            self.update_status("光谱特征分析功能开发中...")
        elif command == "6":
            self.update_status("模型训练功能开发中...")
        elif command == "7":
            self.update_status("模型评估功能开发中...")
        elif command == "8":
            self.update_status("设置功能开发中...")
        elif command.lower() == "q":
            self.exit()
        else:
            self.update_status(f"无效选项: {command}")

    def handle_data_overview_menu(self, command: str) -> None:
        """处理数据概览菜单命令"""
        if command == "0":
            self.current_menu = "main"
            self.refresh_display()
        elif command == "1":
            self.show_basic_stats()
        elif command == "2":
            self.show_dimension_info()
        elif command == "3":
            self.show_memory_usage()
        elif command == "4":
            self.show_data_types()
        else:
            self.update_status(f"无效选项: {command}")

    def handle_class_analysis_menu(self, command: str) -> None:
        """处理类别分析菜单命令"""
        if command == "0":
            self.current_menu = "main"
            self.refresh_display()
        elif command == "1":
            self.show_class_statistics()
        elif command == "2":
            self.update_status("生成类别分布图功能开发中...")
        elif command == "3":
            self.show_class_balance()
        else:
            self.update_status(f"无效选项: {command}")

    def load_dataset(self) -> None:
        """加载数据集"""
        try:
            self.update_status("正在加载数据集...")
            self.visualizer = CropDataVisualizer(data_dir="./dataset")
            
            # 收集数据集信息
            x = self.visualizer.x_data
            y = self.visualizer.y_data
            
            self.dataset_info = {
                "📊 特征数据维度": str(x.shape),
                "🏷️  标签数据维度": str(y.shape),
                "💾 数据类型": f"X: {x.dtype}, Y: {y.dtype}",
                "💿 内存占用": f"{(x.nbytes + y.nbytes) / 1024**2:.2f} MB",
                "📈 值域范围": f"[{x.min():.2f}, {x.max():.2f}]",
                "🎯 类别数量": str(len(np.unique(y))),
                "📦 样本总数": f"{x.shape[0]:,}"
            }
            
            self.data_loaded = True
            self.refresh_display()
            self.update_status("数据集加载成功！")
            
        except FileNotFoundError:
            self.update_status("错误: 未找到数据集文件，请检查 ./dataset 目录")
        except Exception as e:
            self.update_status(f"加载数据集时发生错误: {str(e)}")

    def show_basic_stats(self) -> None:
        """显示基本统计信息"""
        if not self.visualizer:
            return
            
        x = self.visualizer.x_data
        stats_info = f"""
基本统计信息:
  均值: {x.mean():.4f}
  标准差: {x.std():.4f}
  最小值: {x.min():.4f}
  最大值: {x.max():.4f}
  中位数: {np.median(x):.4f}
        """
        
        # 更新信息显示区域
        info_widget = self.query_one("#info", Static)
        info_widget.update(stats_info)
        self.update_status("已显示基本统计信息")

    def show_dimension_info(self) -> None:
        """显示维度信息"""
        if not self.visualizer:
            return
            
        x = self.visualizer.x_data
        y = self.visualizer.y_data
        
        dim_info = f"""
数据维度详情:
  特征数据 (X): {x.shape}
    - 样本数: {x.shape[0]:,}
    - 特征数: {x.shape[1]:,} (如适用)
  标签数据 (Y): {y.shape}
    - 标签数: {y.shape[0]:,}
  数据匹配: {'✓' if x.shape[0] == y.shape[0] else '✗'}
        """
        
        info_widget = self.query_one("#info", Static)
        info_widget.update(dim_info)
        self.update_status("已显示维度信息")

    def show_memory_usage(self) -> None:
        """显示内存使用情况"""
        if not self.visualizer:
            return
            
        x = self.visualizer.x_data
        y = self.visualizer.y_data
        
        memory_info = f"""
内存使用情况:
  特征数据 (X): {x.nbytes / 1024**2:.2f} MB
  标签数据 (Y): {y.nbytes / 1024**2:.2f} MB
  总计: {(x.nbytes + y.nbytes) / 1024**2:.2f} MB
  
  每个样本平均: {(x.nbytes + y.nbytes) / x.shape[0] / 1024:.2f} KB
        """
        
        info_widget = self.query_one("#info", Static)
        info_widget.update(memory_info)
        self.update_status("已显示内存使用情况")

    def show_data_types(self) -> None:
        """显示数据类型信息"""
        if not self.visualizer:
            return
            
        x = self.visualizer.x_data
        y = self.visualizer.y_data
        
        type_info = f"""
数据类型信息:
  特征数据 (X): {x.dtype}
    - 字节大小: {x.itemsize}
    - 数值范围: {np.iinfo(x.dtype) if np.issubdtype(x.dtype, np.integer) else 'N/A'}
  
  标签数据 (Y): {y.dtype}
    - 字节大小: {y.itemsize}
    - 唯一值: {len(np.unique(y))}
        """
        
        info_widget = self.query_one("#info", Static)
        info_widget.update(type_info)
        self.update_status("已显示数据类型信息")

    def show_class_statistics(self) -> None:
        """显示类别统计信息"""
        if not self.visualizer:
            return
            
        y = self.visualizer.y_data
        unique_labels, counts = np.unique(y, return_counts=True)
        
        class_info = "类别统计信息:\n"
        class_info += "=" * 40 + "\n"
        
        for label, count in zip(unique_labels, counts):
            name = self.visualizer.crop_labels.get(label, f"类别{label}")
            percentage = count * 100 / y.size
            class_info += f"  {name:15} | {count:8,} | {percentage:6.2f}%\n"
        
        class_info += "=" * 40 + "\n"
        class_info += f"总计: {y.size:,} 个样本"
        
        info_widget = self.query_one("#info", Static)
        info_widget.update(class_info)
        self.update_status("已显示类别统计信息")

    def show_class_balance(self) -> None:
        """显示类别平衡性分析"""
        if not self.visualizer:
            return
            
        y = self.visualizer.y_data
        unique_labels, counts = np.unique(y, return_counts=True)
        
        # 计算平衡性指标
        max_count = counts.max()
        min_count = counts.min()
        balance_ratio = min_count / max_count
        
        balance_info = f"""
类别平衡性分析:
  最多类别样本数: {max_count:,}
  最少类别样本数: {min_count:,}
  平衡比例: {balance_ratio:.3f}
  
  平衡性评估: {'良好' if balance_ratio > 0.8 else '中等' if balance_ratio > 0.5 else '不平衡'}
  
建议: {'数据分布较为均衡' if balance_ratio > 0.8 else '建议考虑数据增强或重采样技术'}
        """
        
        info_widget = self.query_one("#info", Static)
        info_widget.update(balance_info)
        self.update_status("已显示类别平衡性分析")

    def refresh_display(self) -> None:
        """刷新显示内容"""
        # 更新信息显示区域
        info_widget = self.query_one("#info", Static)
        info_widget.update(self.get_info_display())
        
        # 更新菜单选项区域
        menu_widget = self.query_one("#menu", Static)
        menu_widget.update(self.get_menu_options())

    def update_status(self, message: str) -> None:
        """更新状态栏"""
        status_widget = self.query_one("#status", Static)
        status_widget.update(f"{message} | q=退出")


def main():
    """主函数入口"""
    app = SimpleTUI()
    app.run()


if __name__ == "__main__":
    main()