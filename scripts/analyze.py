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
    parser.add_argument('-i', '--input', required=True, help="输入主文件夹路径，包含多个子文件夹的不同曝光图像")
    parser.add_argument('-o', '--output', required=True, help="输出文件夹路径，包含融合后的图像")
    parser.add_argument('-r', '--report', default='analysis_report.csv', help="分析报告保存路径")
    parser.add_argument('-a', '--analysis_dir', default='analysis_results', help="分析结果（图表等）保存文件夹")
    parser.add_argument('-w', '--workers', type=int, default=None, help="并行处理的最大进程数（默认：CPU核心数-1）")
    parser.add_argument('--cpu_threshold', type=float, default=80.0, help="CPU使用率阈值（百分比，默认80%）")
    parser.add_argument('--mem_threshold', type=float, default=80.0, help="内存使用率阈值（百分比，默认80%）")
    parser.add_argument('--image_type', choices=['original', 'fused', 'both'], default='both', help="选择处理的图像类型：'original'（原图）、'fused'（融合图像）、'both'（全部）")
    parser.add_argument('--only_3d', action='store_true', help="仅生成斜向3D亮度图，跳过频域分析")
    args = parser.parse_args()

    analyzer = HDRAnalyzer(
        input_dir=args.input,
        output_dir=args.output,
        report_path=args.report,
        analysis_dir=args.analysis_dir,
        max_workers=args.workers,
        cpu_threshold=args.cpu_threshold,
        mem_threshold=args.mem_threshold,
        image_type=args.image_type,
        only_3d=args.only_3d
    )
    analyzer.analyze()

if __name__ == "__main__":
    main()
