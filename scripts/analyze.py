# scripts/analyze.py

import argparse
import os
import sys

# 将src目录添加到系统路径，以便导入hdr_analysis模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
src_dir = os.path.join(parent_dir, 'src')
sys.path.append(src_dir)

from hdr_analysis.analyzer import HDRAnalyzer

def main():
    parser = argparse.ArgumentParser(description="分析HDR合成前后的图像")
    parser.add_argument('-i', '--input', required=True, nargs='+', help="要分析的图像文件路径")
    parser.add_argument('-o', '--output', default=None, help="输出主文件夹路径，默认为输入文件夹下的 'output'")
    parser.add_argument('-r', '--report', default=None, help="分析报告保存目录路径")
    parser.add_argument('-a', '--analysis_dir', default=None, help="分析结果（图表等）保存目录路径")
    parser.add_argument('-w', '--workers', type=int, default=None, help="并行处理的最大进程数（默认：CPU核心数-1）")
    parser.add_argument('--cpu_threshold', type=float, default=80.0, help="CPU使用率阈值（百分比，默认80%）")
    parser.add_argument('--mem_threshold', type=float, default=80.0, help="内存使用率阈值（百分比，默认80%）")
    parser.add_argument('--image_type', choices=['original', 'fused', 'both'], default='both', help="选择处理的图像类型：'original'（原图）、'fused'（融合图像）、'both'（全部）")
    parser.add_argument('--statistical', action='store_true', help="执行统计分析")
    parser.add_argument('--frequency', action='store_true', help="执行频域分析")
    parser.add_argument('--difference', action='store_true', help="生成差异图像")
    parser.add_argument('--brightness_3d', action='store_true', help="生成3D亮度图")
    args = parser.parse_args()

    # 设置输出目录
    if args.output:
        output_dir = args.output
    else:
        # 如果输入多个文件，选择第一个文件的目录作为输出目录
        first_input_dir = os.path.dirname(args.input[0])
        output_dir = os.path.join(first_input_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    report_path = os.path.join(args.report, 'analysis_report.csv') if args.report else os.path.join(output_dir, 'analysis_report.csv')
    analysis_dir = args.analysis_dir if args.analysis_dir else os.path.join(output_dir, 'analysis_results')
    os.makedirs(analysis_dir, exist_ok=True)

    # 收集分析选项
    analyses = {
        'statistical': args.statistical,
        'frequency': args.frequency,
        'difference': args.difference,
        'brightness_3d': args.brightness_3d
    }

    if not any(analyses.values()):
        print("请至少选择一个分析选项（--statistical、--frequency、--difference、--brightness_3d）。")
        sys.exit(1)

    # 初始化HDRAnalyzer
    analyzer = HDRAnalyzer(
        selected_images=args.input,
        output_dir=output_dir,
        report_path=report_path,
        analysis_dir=analysis_dir,
        max_workers=args.workers,
        cpu_threshold=args.cpu_threshold,
        mem_threshold=args.mem_threshold,
        image_type=args.image_type,
        analyses=analyses
    )
    analyzer.analyze()

if __name__ == "__main__":
    main()
