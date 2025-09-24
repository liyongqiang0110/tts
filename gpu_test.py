#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU健康状况检查脚本
详细分析显卡资源碎片化、温度、功耗、内存使用等关键指标
"""

import os
import sys
import time
import json
import psutil
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: PyTorch未安装，部分GPU检查功能将不可用")

try:
    import nvidia_ml_py3 as nvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("警告: nvidia-ml-py3未安装，部分NVIDIA GPU检查功能将不可用")

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    print("提示: GPUtil未安装，将使用其他方式获取GPU信息")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'gpu_health_check_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

class GPUHealthChecker:
    """GPU健康状况检查器"""
    
    def __init__(self):
        self.gpu_info = {}
        self.health_report = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {},
            'gpu_details': [],
            'memory_analysis': {},
            'performance_metrics': {},
            'health_issues': [],
            'recommendations': []
        }
        
        # 初始化NVML
        if NVML_AVAILABLE:
            try:
                nvml.nvmlInit()
                self.nvml_initialized = True
                logger.info("NVML初始化成功")
            except Exception as e:
                self.nvml_initialized = False
                logger.warning(f"NVML初始化失败: {e}")
        else:
            self.nvml_initialized = False
    
    def get_system_info(self) -> Dict:
        """获取系统基本信息"""
        logger.info("获取系统信息...")
        
        system_info = {
            'platform': sys.platform,
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'memory_available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
            'python_version': sys.version,
            'cuda_available': TORCH_AVAILABLE and torch.cuda.is_available() if TORCH_AVAILABLE else False
        }
        
        # 获取CUDA版本信息
        if system_info['cuda_available']:
            system_info['cuda_version'] = torch.version.cuda
            system_info['pytorch_version'] = torch.__version__
            system_info['gpu_count'] = torch.cuda.device_count()
        
        # 尝试获取nvidia-smi信息
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                system_info['nvidia_driver_version'] = result.stdout.strip().split('\n')[0]
        except Exception as e:
            logger.warning(f"无法获取NVIDIA驱动版本: {e}")
        
        self.health_report['system_info'] = system_info
        return system_info
    
    def get_gpu_basic_info(self) -> List[Dict]:
        """获取GPU基本信息"""
        logger.info("获取GPU基本信息...")
        gpu_details = []
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_info = {
                    'device_id': i,
                    'name': torch.cuda.get_device_name(i),
                    'capability': torch.cuda.get_device_capability(i),
                    'total_memory_gb': round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2),
                }
                
                # 获取当前内存使用情况
                torch.cuda.set_device(i)
                gpu_info['allocated_memory_gb'] = round(torch.cuda.memory_allocated(i) / (1024**3), 2)
                gpu_info['cached_memory_gb'] = round(torch.cuda.memory_reserved(i) / (1024**3), 2)
                gpu_info['free_memory_gb'] = gpu_info['total_memory_gb'] - gpu_info['cached_memory_gb']
                
                gpu_details.append(gpu_info)
        
        # 使用NVML获取更详细信息
        if self.nvml_initialized:
            try:
                device_count = nvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = nvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # 如果GPU信息已存在，则更新；否则创建新的
                    if i < len(gpu_details):
                        gpu_info = gpu_details[i]
                    else:
                        gpu_info = {'device_id': i}
                        gpu_details.append(gpu_info)
                    
                    # 获取详细信息
                    gpu_info.update({
                        'uuid': nvml.nvmlDeviceGetUUID(handle),
                        'temperature': nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU),
                        'power_usage': nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,  # 转换为瓦特
                        'power_limit': nvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1] / 1000.0,
                        'fan_speed': nvml.nvmlDeviceGetFanSpeed(handle),
                        'gpu_utilization': nvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                        'memory_utilization': nvml.nvmlDeviceGetUtilizationRates(handle).memory,
                    })
                    
                    # 获取内存详细信息
                    try:
                        mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                        gpu_info.update({
                            'nvml_total_memory_gb': round(mem_info.total / (1024**3), 2),
                            'nvml_used_memory_gb': round(mem_info.used / (1024**3), 2),
                            'nvml_free_memory_gb': round(mem_info.free / (1024**3), 2)
                        })
                    except Exception as e:
                        logger.warning(f"获取GPU {i} 内存信息失败: {e}")
            
            except Exception as e:
                logger.error(f"NVML获取GPU信息失败: {e}")
        
        self.health_report['gpu_details'] = gpu_details
        return gpu_details
    
    def analyze_memory_fragmentation(self) -> Dict:
        """分析内存碎片化情况"""
        logger.info("分析内存碎片化情况...")
        
        memory_analysis = {
            'fragmentation_detected': False,
            'fragmentation_level': 'low',
            'memory_efficiency': {},
            'allocation_patterns': []
        }
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                
                # 获取详细内存统计
                try:
                    memory_stats = torch.cuda.memory_stats(i)
                    total_memory = torch.cuda.get_device_properties(i).total_memory
                    allocated = torch.cuda.memory_allocated(i)
                    reserved = torch.cuda.memory_reserved(i)
                    
                    # 计算碎片化指标
                    if reserved > 0:
                        allocation_ratio = allocated / reserved
                        memory_efficiency = allocated / total_memory
                        
                        device_analysis = {
                            'device_id': i,
                            'allocation_ratio': round(allocation_ratio, 3),
                            'memory_efficiency': round(memory_efficiency, 3),
                            'reserved_gb': round(reserved / (1024**3), 2),
                            'allocated_gb': round(allocated / (1024**3), 2),
                            'total_gb': round(total_memory / (1024**3), 2)
                        }
                        
                        # 判断碎片化程度
                        if allocation_ratio < 0.7:  # 保留内存中实际使用不足70%
                            device_analysis['fragmentation_detected'] = True
                            if allocation_ratio < 0.5:
                                device_analysis['fragmentation_level'] = 'high'
                            elif allocation_ratio < 0.6:
                                device_analysis['fragmentation_level'] = 'medium'
                            else:
                                device_analysis['fragmentation_level'] = 'low'
                        else:
                            device_analysis['fragmentation_detected'] = False
                            device_analysis['fragmentation_level'] = 'none'
                        
                        # 获取更详细的内存统计
                        if 'num_alloc_retries' in memory_stats:
                            device_analysis['allocation_retries'] = memory_stats['num_alloc_retries']
                        if 'num_ooms' in memory_stats:
                            device_analysis['out_of_memory_errors'] = memory_stats['num_ooms']
                        
                        memory_analysis['memory_efficiency'][f'gpu_{i}'] = device_analysis
                        
                        # 更新全局碎片化状态
                        if device_analysis['fragmentation_detected']:
                            memory_analysis['fragmentation_detected'] = True
                            if device_analysis['fragmentation_level'] == 'high':
                                memory_analysis['fragmentation_level'] = 'high'
                            elif device_analysis['fragmentation_level'] == 'medium' and memory_analysis['fragmentation_level'] != 'high':
                                memory_analysis['fragmentation_level'] = 'medium'
                
                except Exception as e:
                    logger.warning(f"分析GPU {i} 内存碎片化失败: {e}")
        
        self.health_report['memory_analysis'] = memory_analysis
        return memory_analysis
    
    def stress_test_memory_allocation(self, test_duration: int = 30) -> Dict:
        """进行内存分配压力测试"""
        logger.info(f"开始内存分配压力测试（持续{test_duration}秒）...")
        
        stress_results = {
            'test_duration': test_duration,
            'allocation_success_rate': {},
            'performance_degradation': {},
            'memory_leak_detected': False
        }
        
        if not (TORCH_AVAILABLE and torch.cuda.is_available()):
            logger.warning("CUDA不可用，跳过压力测试")
            return stress_results
        
        for gpu_id in range(torch.cuda.device_count()):
            torch.cuda.set_device(gpu_id)
            logger.info(f"测试GPU {gpu_id}...")
            
            initial_memory = torch.cuda.memory_allocated(gpu_id)
            successful_allocs = 0
            failed_allocs = 0
            allocation_times = []
            
            start_time = time.time()
            
            while time.time() - start_time < test_duration:
                try:
                    # 尝试分配不同大小的内存块
                    alloc_start = time.time()
                    
                    # 随机分配100MB到500MB的内存
                    size_mb = 100 + (successful_allocs % 5) * 100
                    tensor = torch.randn(size_mb * 1024 * 256, device=f'cuda:{gpu_id}')
                    
                    alloc_time = time.time() - alloc_start
                    allocation_times.append(alloc_time)
                    
                    # 立即释放内存
                    del tensor
                    torch.cuda.empty_cache()
                    
                    successful_allocs += 1
                    time.sleep(0.1)
                    
                except torch.cuda.OutOfMemoryError:
                    failed_allocs += 1
                    torch.cuda.empty_cache()
                    time.sleep(0.5)
                except Exception as e:
                    logger.warning(f"内存分配测试异常: {e}")
                    failed_allocs += 1
                    time.sleep(0.1)
            
            # 分析结果
            total_attempts = successful_allocs + failed_allocs
            success_rate = successful_allocs / total_attempts if total_attempts > 0 else 0
            avg_alloc_time = sum(allocation_times) / len(allocation_times) if allocation_times else 0
            
            final_memory = torch.cuda.memory_allocated(gpu_id)
            memory_leak = final_memory > initial_memory + 100 * 1024 * 1024  # 超过100MB认为可能有内存泄漏
            
            stress_results['allocation_success_rate'][f'gpu_{gpu_id}'] = {
                'success_rate': round(success_rate, 3),
                'successful_allocations': successful_allocs,
                'failed_allocations': failed_allocs,
                'average_allocation_time_ms': round(avg_alloc_time * 1000, 2),
                'memory_leak_suspected': memory_leak,
                'initial_memory_gb': round(initial_memory / (1024**3), 3),
                'final_memory_gb': round(final_memory / (1024**3), 3)
            }
            
            if memory_leak:
                stress_results['memory_leak_detected'] = True
        
        return stress_results
    
    def check_thermal_status(self) -> Dict:
        """检查热状态"""
        logger.info("检查GPU热状态...")
        
        thermal_status = {
            'overheating_detected': False,
            'thermal_throttling': False,
            'temperature_readings': {}
        }
        
        if self.nvml_initialized:
            try:
                device_count = nvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = nvml.nvmlDeviceGetHandleByIndex(i)
                    
                    temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                    
                    # 获取温度阈值
                    try:
                        temp_threshold = nvml.nvmlDeviceGetTemperatureThreshold(handle, nvml.NVML_TEMPERATURE_THRESHOLD_SHUTDOWN)
                        thermal_status['temperature_readings'][f'gpu_{i}'] = {
                            'current_temp': temp,
                            'shutdown_threshold': temp_threshold,
                            'overheating': temp > temp_threshold * 0.85,  # 超过85%阈值认为过热
                            'thermal_throttling': temp > temp_threshold * 0.95  # 超过95%阈值认为热降频
                        }
                    except:
                        # 如果无法获取阈值，使用经验值
                        thermal_status['temperature_readings'][f'gpu_{i}'] = {
                            'current_temp': temp,
                            'overheating': temp > 80,  # 超过80度认为过热
                            'thermal_throttling': temp > 90  # 超过90度认为热降频
                        }
                    
                    if thermal_status['temperature_readings'][f'gpu_{i}']['overheating']:
                        thermal_status['overheating_detected'] = True
                    
                    if thermal_status['temperature_readings'][f'gpu_{i}']['thermal_throttling']:
                        thermal_status['thermal_throttling'] = True
            
            except Exception as e:
                logger.error(f"检查热状态失败: {e}")
        
        return thermal_status
    
    def diagnose_issues(self) -> List[str]:
        """诊断问题并提供建议"""
        logger.info("诊断GPU问题...")
        
        issues = []
        recommendations = []
        
        # 检查内存碎片化
        memory_analysis = self.health_report.get('memory_analysis', {})
        if memory_analysis.get('fragmentation_detected', False):
            level = memory_analysis.get('fragmentation_level', 'unknown')
            issues.append(f"检测到GPU内存碎片化，碎片化程度: {level}")
            
            if level == 'high':
                recommendations.extend([
                    "建议立即重启GPU进程或清理内存缓存",
                    "考虑使用torch.cuda.empty_cache()定期清理内存",
                    "检查代码中是否存在内存泄漏",
                    "考虑减少单次内存分配的大小"
                ])
            elif level == 'medium':
                recommendations.extend([
                    "建议定期清理GPU内存缓存",
                    "监控内存使用模式，避免频繁的大块内存分配"
                ])
        
        # 检查温度问题
        thermal_status = self.check_thermal_status()
        if thermal_status.get('overheating_detected', False):
            issues.append("检测到GPU过热")
            recommendations.extend([
                "检查GPU散热系统是否正常工作",
                "清理GPU风扇和散热片上的灰尘",
                "检查机箱通风是否良好",
                "考虑降低GPU功耗限制"
            ])
        
        if thermal_status.get('thermal_throttling', False):
            issues.append("检测到GPU热降频")
            recommendations.append("GPU因过热而降频，严重影响性能，需要立即改善散热")
        
        # 检查系统资源
        system_info = self.health_report.get('system_info', {})
        memory_available = system_info.get('memory_available_gb', 0)
        memory_total = system_info.get('memory_total_gb', 0)
        
        if memory_available / memory_total < 0.2:  # 可用内存低于20%
            issues.append("系统内存不足")
            recommendations.extend([
                "释放系统内存",
                "考虑增加系统内存",
                "检查是否有内存泄漏的进程"
            ])
        
        # 检查GPU数量和可用性
        if not system_info.get('cuda_available', False):
            issues.append("CUDA不可用")
            recommendations.extend([
                "检查NVIDIA驱动是否正确安装",
                "检查CUDA工具包是否正确安装",
                "检查PyTorch是否支持当前的CUDA版本"
            ])
        
        self.health_report['health_issues'] = issues
        self.health_report['recommendations'] = recommendations
        
        return issues
    
    def generate_report(self, save_to_file: bool = True) -> Dict:
        """生成完整的健康检查报告"""
        logger.info("生成GPU健康检查报告...")
        
        # 执行所有检查
        self.get_system_info()
        self.get_gpu_basic_info()
        self.analyze_memory_fragmentation()
        
        # 执行压力测试（可选）
        try:
            stress_results = self.stress_test_memory_allocation(10)  # 短测试
            self.health_report['stress_test'] = stress_results
        except Exception as e:
            logger.warning(f"压力测试失败: {e}")
        
        # 添加其他分析
        self.health_report['thermal_status'] = self.check_thermal_status()
        
        # 诊断问题
        self.diagnose_issues()
        
        # 添加性能指标
        self.health_report['performance_metrics'] = self.calculate_performance_metrics()
        
        # 保存报告到文件
        if save_to_file:
            filename = f"gpu_health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.health_report, f, indent=2, ensure_ascii=False)
                logger.info(f"健康检查报告已保存到: {filename}")
            except Exception as e:
                logger.error(f"保存报告失败: {e}")
        
        return self.health_report
    
    def calculate_performance_metrics(self) -> Dict:
        """计算性能指标"""
        metrics = {
            'overall_health_score': 100,
            'memory_health_score': 100,
            'thermal_health_score': 100
        }
        
        # 基于各种问题降低健康分数
        memory_analysis = self.health_report.get('memory_analysis', {})
        if memory_analysis.get('fragmentation_detected', False):
            level = memory_analysis.get('fragmentation_level', 'low')
            if level == 'high':
                metrics['memory_health_score'] -= 40
            elif level == 'medium':
                metrics['memory_health_score'] -= 20
            else:
                metrics['memory_health_score'] -= 10
        
        thermal_status = self.health_report.get('thermal_status', {})
        if thermal_status.get('overheating_detected', False):
            metrics['thermal_health_score'] -= 30
        if thermal_status.get('thermal_throttling', False):
            metrics['thermal_health_score'] -= 50
        
        # 计算总体健康分数
        metrics['overall_health_score'] = round(
            (metrics['memory_health_score'] + metrics['thermal_health_score']) / 2
        )
        
        return metrics
    
    def print_summary(self):
        """打印检查结果摘要"""
        print("\n" + "="*60)
        print("              GPU健康状况检查报告")
        print("="*60)
        
        # 系统信息
        system_info = self.health_report.get('system_info', {})
        print(f"\n【系统信息】")
        print(f"CUDA可用: {system_info.get('cuda_available', False)}")
        print(f"GPU数量: {system_info.get('gpu_count', 0)}")
        print(f"系统内存: {system_info.get('memory_available_gb', 0):.1f}GB / {system_info.get('memory_total_gb', 0):.1f}GB")
        if 'nvidia_driver_version' in system_info:
            print(f"NVIDIA驱动版本: {system_info['nvidia_driver_version']}")
        
        # GPU详细信息
        gpu_details = self.health_report.get('gpu_details', [])
        for gpu in gpu_details:
            print(f"\n【GPU {gpu.get('device_id', 'N/A')}】")
            print(f"型号: {gpu.get('name', 'Unknown')}")
            print(f"显存: {gpu.get('free_memory_gb', 0):.1f}GB可用 / {gpu.get('total_memory_gb', 0):.1f}GB总计")
            if 'temperature' in gpu:
                print(f"温度: {gpu['temperature']}°C")
            if 'gpu_utilization' in gpu:
                print(f"GPU使用率: {gpu['gpu_utilization']}%")
            if 'power_usage' in gpu:
                print(f"功耗: {gpu['power_usage']:.1f}W")
        
        # 内存分析
        memory_analysis = self.health_report.get('memory_analysis', {})
        print(f"\n【内存分析】")
        fragmentation_status = '是' if memory_analysis.get('fragmentation_detected', False) else '否'
        print(f"碎片化检测: {fragmentation_status}")
        if memory_analysis.get('fragmentation_detected', False):
            print(f"碎片化程度: {memory_analysis.get('fragmentation_level', 'unknown')}")
        
        # 热状态
        thermal_status = self.health_report.get('thermal_status', {})
        print(f"\n【热状态】")
        overheating_status = '是' if thermal_status.get('overheating_detected', False) else '否'
        print(f"过热检测: {overheating_status}")
        throttling_status = '是' if thermal_status.get('thermal_throttling', False) else '否'
        print(f"热降频检测: {throttling_status}")
        
        # 性能指标
        performance_metrics = self.health_report.get('performance_metrics', {})
        print(f"\n【性能指标】")
        print(f"总体健康分数: {performance_metrics.get('overall_health_score', 0)}")
        print(f"内存健康分数: {performance_metrics.get('memory_health_score', 0)}")
        print(f"热状态健康分数: {performance_metrics.get('thermal_health_score', 0)}")
        
        # 问题和建议
        issues = self.health_report.get('health_issues', [])
        recommendations = self.health_report.get('recommendations', [])
        
        if issues:
            print(f"\n【发现的问题】")
            for i, issue in enumerate(issues, 1):
                print(f"{i}. {issue}")
        
        if recommendations:
            print(f"\n【建议措施】")
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")
        
        print("\n" + "="*60)
        print("检查完成！详细报告已保存到JSON文件。")
        print("="*60)

def main():
    """主函数"""
    print("启动GPU健康状况检查器...")
    
    # 创建检查器实例
    checker = GPUHealthChecker()
    
    try:
        # 生成完整报告
        report = checker.generate_report()
        
        # 打印摘要
        checker.print_summary()
        
        # 特别检查重要问题
        critical_issues = []
        
        # 检查高度碎片化
        memory_analysis = report.get('memory_analysis', {})
        if memory_analysis.get('fragmentation_level') == 'high':
            critical_issues.append("严重内存碎片化")
        
        # 检查过热
        thermal_status = report.get('thermal_status', {})
        if thermal_status.get('thermal_throttling', False):
            critical_issues.append("GPU热降频")
        
        # 检查CUDA可用性
        system_info = report.get('system_info', {})
        if not system_info.get('cuda_available', False):
            critical_issues.append("CUDA不可用")
        
        if critical_issues:
            print(f"\n⚠️  关键问题警告:")
            for issue in critical_issues:
                print(f"   - {issue}")
            print("\n建议立即处理这些问题以确保服务正常运行。")
        else:
            print("\n✅ 没有发现关键问题，GPU状态良好。")
    
    except Exception as e:
        logger.error(f"GPU健康检查失败: {e}")
        print(f"检查过程中发生错误: {e}")
        
if __name__ == "__main__":
    main()