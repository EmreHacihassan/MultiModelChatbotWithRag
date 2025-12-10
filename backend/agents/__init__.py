"""
AI Agents - LangChain/LangGraph Powered Intelligent Assistants
==============================================================

Bu modül, araç kullanan, karar veren ve çok adımlı düşünme yapabilen
akıllı AI asistanlarını içerir.

Özellikler:
- Tool Calling: Web arama, hesap makinesi, kod çalıştırma
- Multi-step Reasoning: Karmaşık soruları adım adım çözme
- ReAct Pattern: Reasoning + Acting döngüsü
- Memory: Konuşma geçmişi takibi

Kullanım:
    from backend.agents import AgentExecutor, get_agent
    
    agent = get_agent("react")
    result = await agent.run("Python ile fibonacci hesapla")
"""

import os
import re
import json
import asyncio
import logging
import subprocess
import tempfile
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger('agents')

# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

class ToolResult:
    """Tool çalıştırma sonucu."""
    
    def __init__(self, success: bool, output: str, tool_name: str, 
                 execution_time: float = 0, metadata: Dict = None):
        self.success = success
        self.output = output
        self.tool_name = tool_name
        self.execution_time = execution_time
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'output': self.output,
            'tool_name': self.tool_name,
            'execution_time': self.execution_time,
            'metadata': self.metadata
        }


class BaseTool(ABC):
    """Tüm tool'ların base class'ı."""
    
    name: str = "base_tool"
    description: str = "Base tool description"
    parameters: Dict[str, Any] = {}
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Tool'u çalıştır."""
        pass
    
    def get_schema(self) -> Dict:
        """OpenAI function calling formatında schema döndür."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.parameters,
                "required": list(self.parameters.keys())
            }
        }


# =============================================================================
# BUILT-IN TOOLS
# =============================================================================

class CalculatorTool(BaseTool):
    """Matematiksel hesaplamalar yapan tool."""
    
    name = "calculator"
    description = """Matematiksel hesaplamalar yapar. Temel aritmetik (+, -, *, /), 
    üs alma (**), karekök, trigonometri ve daha fazlasını destekler.
    Örnek: '2 + 2', 'sqrt(16)', 'sin(3.14159/2)', '2**10'"""
    
    parameters = {
        "expression": {
            "type": "string",
            "description": "Hesaplanacak matematiksel ifade"
        }
    }
    
    # Güvenli math fonksiyonları
    SAFE_FUNCTIONS = {
        'abs': abs, 'round': round, 'min': min, 'max': max,
        'sum': sum, 'pow': pow, 'len': len,
    }
    
    def __init__(self):
        # Math modülünden fonksiyonları ekle
        import math
        self.SAFE_FUNCTIONS.update({
            'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos, 
            'tan': math.tan, 'log': math.log, 'log10': math.log10,
            'exp': math.exp, 'pi': math.pi, 'e': math.e,
            'ceil': math.ceil, 'floor': math.floor,
            'factorial': math.factorial, 'gcd': math.gcd,
            'degrees': math.degrees, 'radians': math.radians,
        })
    
    async def execute(self, expression: str) -> ToolResult:
        """Matematiksel ifadeyi hesapla."""
        start_time = datetime.now()
        
        try:
            # Güvenlik: sadece izin verilen karakterler
            allowed_chars = set('0123456789+-*/.()[], ')
            expr_chars = set(expression.replace(' ', ''))
            
            # Fonksiyon isimlerini kontrol et
            for func_name in self.SAFE_FUNCTIONS.keys():
                expr_chars -= set(func_name)
            
            # Hala kalan karakterler varsa ve izin verilmemişse reddet
            dangerous = expr_chars - allowed_chars
            
            # Basit güvenlik kontrolü: __builtins__ gibi tehlikeli şeyleri engelle
            if '__' in expression or 'import' in expression or 'eval' in expression:
                return ToolResult(
                    success=False,
                    output="Güvenlik hatası: İzin verilmeyen ifade",
                    tool_name=self.name
                )
            
            # Hesapla
            result = eval(expression, {"__builtins__": {}}, self.SAFE_FUNCTIONS)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ToolResult(
                success=True,
                output=f"{expression} = {result}",
                tool_name=self.name,
                execution_time=execution_time,
                metadata={'expression': expression, 'result': result}
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                output=f"Hesaplama hatası: {str(e)}",
                tool_name=self.name
            )


class PythonExecutorTool(BaseTool):
    """Python kodu çalıştıran tool (sandboxed)."""
    
    name = "python_executor"
    description = """Python kodu çalıştırır ve sonucu döndürür. 
    Veri işleme, algoritma testi, hesaplamalar için kullanılabilir.
    Kod GÜVENLI bir sandbox ortamında çalışır."""
    
    parameters = {
        "code": {
            "type": "string",
            "description": "Çalıştırılacak Python kodu"
        }
    }
    
    # Tehlikeli modüller/fonksiyonlar
    BLACKLIST = [
        'os.system', 'subprocess', 'eval', 'exec', '__import__',
        'open(', 'file(', 'input(', 'raw_input',
        'compile', 'globals', 'locals', 'vars',
        'getattr', 'setattr', 'delattr',
        'shutil', 'pathlib', 'glob',
    ]
    
    async def execute(self, code: str) -> ToolResult:
        """Python kodunu güvenli ortamda çalıştır."""
        start_time = datetime.now()
        
        # Güvenlik kontrolü
        code_lower = code.lower()
        for dangerous in self.BLACKLIST:
            if dangerous.lower() in code_lower:
                return ToolResult(
                    success=False,
                    output=f"Güvenlik hatası: '{dangerous}' kullanımı yasak",
                    tool_name=self.name
                )
        
        try:
            # Geçici dosya oluştur
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                # Güvenli wrapper ekle
                safe_code = f"""
import sys
import math
import json
import random
import statistics
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from itertools import combinations, permutations

# Sonuç yakalama
_result = None

{code}

# Son değişkeni yazdır
if '_result' in dir():
    print(_result)
"""
                f.write(safe_code)
                temp_file = f.name
            
            # Subprocess ile çalıştır (timeout ile)
            try:
                result = subprocess.run(
                    ['python', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=10,  # 10 saniye timeout
                    cwd=tempfile.gettempdir()
                )
                
                output = result.stdout.strip()
                if result.stderr:
                    output += f"\n[Stderr]: {result.stderr.strip()}"
                
                if result.returncode != 0:
                    return ToolResult(
                        success=False,
                        output=f"Kod hatası:\n{output}",
                        tool_name=self.name
                    )
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                return ToolResult(
                    success=True,
                    output=output if output else "(Çıktı yok)",
                    tool_name=self.name,
                    execution_time=execution_time
                )
                
            except subprocess.TimeoutExpired:
                return ToolResult(
                    success=False,
                    output="Timeout: Kod 10 saniyeden fazla sürdü",
                    tool_name=self.name
                )
            finally:
                # Geçici dosyayı sil
                try:
                    os.unlink(temp_file)
                except:
                    pass
                    
        except Exception as e:
            return ToolResult(
                success=False,
                output=f"Çalıştırma hatası: {str(e)}",
                tool_name=self.name
            )


class WebSearchTool(BaseTool):
    """Web araması yapan tool."""
    
    name = "web_search"
    description = """İnternette arama yapar ve sonuçları döndürür.
    Güncel bilgiler, haberler, araştırma için kullanılır."""
    
    parameters = {
        "query": {
            "type": "string",
            "description": "Arama sorgusu"
        }
    }
    
    async def execute(self, query: str) -> ToolResult:
        """Web araması yap."""
        start_time = datetime.now()
        
        try:
            # DuckDuckGo API kullan (ücretsiz, API key gerektirmez)
            import httpx
            
            async with httpx.AsyncClient(timeout=15) as client:
                # DuckDuckGo Instant Answer API
                response = await client.get(
                    "https://api.duckduckgo.com/",
                    params={
                        "q": query,
                        "format": "json",
                        "no_html": 1,
                        "skip_disambig": 1
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    results = []
                    
                    # Abstract (özet bilgi)
                    if data.get('Abstract'):
                        results.append(f"📝 Özet: {data['Abstract']}")
                        if data.get('AbstractSource'):
                            results.append(f"   Kaynak: {data['AbstractSource']}")
                    
                    # Answer (direkt cevap)
                    if data.get('Answer'):
                        results.append(f"✅ Cevap: {data['Answer']}")
                    
                    # Definition
                    if data.get('Definition'):
                        results.append(f"📖 Tanım: {data['Definition']}")
                    
                    # Related topics
                    related = data.get('RelatedTopics', [])[:5]
                    if related:
                        results.append("\n🔗 İlgili Konular:")
                        for topic in related:
                            if isinstance(topic, dict) and topic.get('Text'):
                                results.append(f"  • {topic['Text'][:200]}...")
                    
                    if results:
                        output = "\n".join(results)
                    else:
                        output = f"'{query}' için direkt sonuç bulunamadı. Daha spesifik bir arama deneyin."
                    
                    execution_time = (datetime.now() - start_time).total_seconds()
                    
                    return ToolResult(
                        success=True,
                        output=output,
                        tool_name=self.name,
                        execution_time=execution_time,
                        metadata={'query': query}
                    )
                else:
                    return ToolResult(
                        success=False,
                        output=f"Arama hatası: HTTP {response.status_code}",
                        tool_name=self.name
                    )
                    
        except ImportError:
            return ToolResult(
                success=False,
                output="httpx yüklü değil. pip install httpx",
                tool_name=self.name
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=f"Arama hatası: {str(e)}",
                tool_name=self.name
            )


class DateTimeTool(BaseTool):
    """Tarih ve saat işlemleri yapan tool."""
    
    name = "datetime"
    description = """Tarih ve saat bilgisi verir, tarih hesaplamaları yapar.
    Örnek: 'now', '2024-01-01', 'add 7 days', 'diff 2024-01-01 2024-12-31'"""
    
    parameters = {
        "operation": {
            "type": "string",
            "description": "İşlem: 'now', 'parse DATE', 'add N days/hours', 'diff DATE1 DATE2'"
        }
    }
    
    async def execute(self, operation: str) -> ToolResult:
        """Tarih/saat işlemi yap."""
        from datetime import datetime, timedelta
        
        try:
            op = operation.lower().strip()
            
            if op == 'now':
                now = datetime.now()
                output = f"""📅 Şu anki tarih ve saat:
  • Tarih: {now.strftime('%Y-%m-%d')}
  • Saat: {now.strftime('%H:%M:%S')}
  • Gün: {now.strftime('%A')}
  • Hafta: {now.isocalendar()[1]}
  • Unix timestamp: {int(now.timestamp())}"""
                
            elif op.startswith('add '):
                # "add 7 days" veya "add 3 hours"
                parts = op.split()
                if len(parts) >= 3:
                    amount = int(parts[1])
                    unit = parts[2].rstrip('s')  # days -> day
                    
                    now = datetime.now()
                    if unit == 'day':
                        future = now + timedelta(days=amount)
                    elif unit == 'hour':
                        future = now + timedelta(hours=amount)
                    elif unit == 'minute':
                        future = now + timedelta(minutes=amount)
                    elif unit == 'week':
                        future = now + timedelta(weeks=amount)
                    else:
                        return ToolResult(False, f"Bilinmeyen birim: {unit}", self.name)
                    
                    output = f"📅 {amount} {parts[2]} sonra: {future.strftime('%Y-%m-%d %H:%M:%S')}"
                else:
                    output = "Hatalı format. Örnek: 'add 7 days'"
                    
            elif op.startswith('diff '):
                # "diff 2024-01-01 2024-12-31"
                parts = op.split()
                if len(parts) >= 3:
                    date1 = datetime.strptime(parts[1], '%Y-%m-%d')
                    date2 = datetime.strptime(parts[2], '%Y-%m-%d')
                    diff = abs((date2 - date1).days)
                    output = f"📅 İki tarih arası: {diff} gün"
                else:
                    output = "Hatalı format. Örnek: 'diff 2024-01-01 2024-12-31'"
                    
            else:
                # Tarih parse etmeye çalış
                try:
                    parsed = datetime.strptime(op, '%Y-%m-%d')
                    now = datetime.now()
                    diff = (parsed - now).days
                    status = "geçmişte" if diff < 0 else "gelecekte"
                    output = f"📅 {op}: {abs(diff)} gün {status}"
                except:
                    output = f"Bilinmeyen işlem: {operation}"
            
            return ToolResult(success=True, output=output, tool_name=self.name)
            
        except Exception as e:
            return ToolResult(success=False, output=f"Hata: {str(e)}", tool_name=self.name)


class WikipediaTool(BaseTool):
    """Wikipedia'dan bilgi çeken tool."""
    
    name = "wikipedia"
    description = """Wikipedia'dan bilgi arar ve özet döndürür.
    Genel bilgi, biyografi, tarih, bilim konuları için kullanılır."""
    
    parameters = {
        "topic": {
            "type": "string",
            "description": "Aranacak konu"
        }
    }
    
    async def execute(self, topic: str) -> ToolResult:
        """Wikipedia'dan bilgi al."""
        try:
            import httpx
            
            async with httpx.AsyncClient(timeout=15) as client:
                # Wikipedia API
                response = await client.get(
                    "https://en.wikipedia.org/api/rest_v1/page/summary/" + topic.replace(' ', '_')
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    title = data.get('title', topic)
                    extract = data.get('extract', 'Bilgi bulunamadı')
                    
                    output = f"""📚 {title}

{extract[:1500]}...

🔗 Daha fazla: https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"""
                    
                    return ToolResult(
                        success=True,
                        output=output,
                        tool_name=self.name,
                        metadata={'title': title}
                    )
                else:
                    # Türkçe Wikipedia dene
                    response = await client.get(
                        "https://tr.wikipedia.org/api/rest_v1/page/summary/" + topic.replace(' ', '_')
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        return ToolResult(
                            success=True,
                            output=f"📚 {data.get('title')}\n\n{data.get('extract', '')}",
                            tool_name=self.name
                        )
                    
                    return ToolResult(
                        success=False,
                        output=f"'{topic}' Wikipedia'da bulunamadı",
                        tool_name=self.name
                    )
                    
        except Exception as e:
            return ToolResult(success=False, output=f"Hata: {str(e)}", tool_name=self.name)


class JSONParserTool(BaseTool):
    """JSON verilerini işleyen tool."""
    
    name = "json_parser"
    description = """JSON verilerini parse eder, formatlar veya query yapar.
    API yanıtlarını işlemek, veri çıkarmak için kullanılır."""
    
    parameters = {
        "json_data": {
            "type": "string",
            "description": "JSON string veya işlem: 'format', 'keys', 'get path.to.value'"
        }
    }
    
    async def execute(self, json_data: str) -> ToolResult:
        """JSON işle."""
        try:
            # Önce JSON parse et
            data = json.loads(json_data)
            
            # Güzel formatla
            formatted = json.dumps(data, indent=2, ensure_ascii=False)
            
            output = f"✅ Geçerli JSON ({len(str(data))} karakter)\n\n{formatted[:2000]}"
            
            return ToolResult(
                success=True,
                output=output,
                tool_name=self.name,
                metadata={'keys': list(data.keys()) if isinstance(data, dict) else None}
            )
            
        except json.JSONDecodeError as e:
            return ToolResult(
                success=False,
                output=f"JSON parse hatası: {str(e)}",
                tool_name=self.name
            )


class TextAnalyzerTool(BaseTool):
    """Metin analizi yapan tool."""
    
    name = "text_analyzer"
    description = """Metin analizi yapar: kelime sayısı, karakter sayısı, 
    en sık kelimeler, duygu analizi tahmini."""
    
    parameters = {
        "text": {
            "type": "string",
            "description": "Analiz edilecek metin"
        }
    }
    
    async def execute(self, text: str) -> ToolResult:
        """Metin analizi yap."""
        from collections import Counter
        import re
        
        try:
            # Temel istatistikler
            char_count = len(text)
            word_count = len(text.split())
            sentence_count = len(re.findall(r'[.!?]+', text)) or 1
            
            # Kelime frekansı
            words = re.findall(r'\b\w+\b', text.lower())
            # Stop words (basit)
            stop_words = {'ve', 'bir', 'bu', 'da', 'de', 'için', 'ile', 'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for'}
            filtered_words = [w for w in words if w not in stop_words and len(w) > 2]
            word_freq = Counter(filtered_words).most_common(10)
            
            # Basit duygu analizi (keyword-based)
            positive_words = {'güzel', 'harika', 'mükemmel', 'iyi', 'başarılı', 'mutlu', 'great', 'good', 'excellent', 'happy', 'love', 'amazing'}
            negative_words = {'kötü', 'berbat', 'üzgün', 'başarısız', 'sorun', 'hata', 'bad', 'terrible', 'sad', 'hate', 'problem', 'error'}
            
            pos_count = sum(1 for w in words if w in positive_words)
            neg_count = sum(1 for w in words if w in negative_words)
            
            if pos_count > neg_count:
                sentiment = "😊 Pozitif"
            elif neg_count > pos_count:
                sentiment = "😔 Negatif"
            else:
                sentiment = "😐 Nötr"
            
            output = f"""📊 Metin Analizi Sonuçları:

📝 Temel İstatistikler:
  • Karakter sayısı: {char_count:,}
  • Kelime sayısı: {word_count:,}
  • Cümle sayısı: {sentence_count}
  • Ort. kelime/cümle: {word_count/sentence_count:.1f}

🔤 En Sık Kelimeler:
{chr(10).join(f'  • {word}: {count}' for word, count in word_freq[:5])}

💭 Duygu Tahmini: {sentiment}
  (Pozitif kelime: {pos_count}, Negatif kelime: {neg_count})
"""
            
            return ToolResult(
                success=True,
                output=output,
                tool_name=self.name,
                metadata={
                    'char_count': char_count,
                    'word_count': word_count,
                    'sentiment': sentiment
                }
            )
            
        except Exception as e:
            return ToolResult(success=False, output=f"Analiz hatası: {str(e)}", tool_name=self.name)


class UnitConverterTool(BaseTool):
    """Birim dönüştürme tool'u."""
    
    name = "unit_converter"
    description = """Birim dönüşümü yapar: uzunluk, ağırlık, sıcaklık, hacim.
    Örnek: '100 km to miles', '25 celsius to fahrenheit', '5 kg to pounds'"""
    
    parameters = {
        "conversion": {
            "type": "string",
            "description": "Dönüşüm ifadesi: 'VALUE FROM_UNIT to TO_UNIT'"
        }
    }
    
    CONVERSIONS = {
        # Uzunluk
        ('km', 'miles'): lambda x: x * 0.621371,
        ('miles', 'km'): lambda x: x * 1.60934,
        ('m', 'feet'): lambda x: x * 3.28084,
        ('feet', 'm'): lambda x: x * 0.3048,
        ('cm', 'inch'): lambda x: x * 0.393701,
        ('inch', 'cm'): lambda x: x * 2.54,
        
        # Ağırlık
        ('kg', 'pounds'): lambda x: x * 2.20462,
        ('pounds', 'kg'): lambda x: x * 0.453592,
        ('g', 'oz'): lambda x: x * 0.035274,
        ('oz', 'g'): lambda x: x * 28.3495,
        
        # Sıcaklık
        ('celsius', 'fahrenheit'): lambda x: (x * 9/5) + 32,
        ('fahrenheit', 'celsius'): lambda x: (x - 32) * 5/9,
        ('celsius', 'kelvin'): lambda x: x + 273.15,
        ('kelvin', 'celsius'): lambda x: x - 273.15,
        
        # Hacim
        ('liter', 'gallon'): lambda x: x * 0.264172,
        ('gallon', 'liter'): lambda x: x * 3.78541,
        
        # Veri
        ('mb', 'gb'): lambda x: x / 1024,
        ('gb', 'mb'): lambda x: x * 1024,
        ('gb', 'tb'): lambda x: x / 1024,
        ('tb', 'gb'): lambda x: x * 1024,
    }
    
    async def execute(self, conversion: str) -> ToolResult:
        """Birim dönüşümü yap."""
        try:
            # Parse: "100 km to miles"
            pattern = r'([\d.]+)\s*(\w+)\s+to\s+(\w+)'
            match = re.match(pattern, conversion.lower().strip())
            
            if not match:
                return ToolResult(
                    success=False,
                    output="Hatalı format. Örnek: '100 km to miles'",
                    tool_name=self.name
                )
            
            value = float(match.group(1))
            from_unit = match.group(2)
            to_unit = match.group(3)
            
            # Dönüşüm fonksiyonunu bul
            key = (from_unit, to_unit)
            if key in self.CONVERSIONS:
                result = self.CONVERSIONS[key](value)
                output = f"📐 {value} {from_unit} = {result:.4f} {to_unit}"
            else:
                available = ', '.join([f"{f}→{t}" for f, t in self.CONVERSIONS.keys()])
                output = f"Desteklenmeyen dönüşüm: {from_unit} → {to_unit}\n\nDesteklenen: {available}"
                return ToolResult(success=False, output=output, tool_name=self.name)
            
            return ToolResult(
                success=True,
                output=output,
                tool_name=self.name,
                metadata={'from': from_unit, 'to': to_unit, 'result': result}
            )
            
        except Exception as e:
            return ToolResult(success=False, output=f"Dönüşüm hatası: {str(e)}", tool_name=self.name)


# =============================================================================
# TOOL REGISTRY
# =============================================================================

class ToolRegistry:
    """Tool kayıt ve yönetim sistemi."""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._register_builtin_tools()
    
    def _register_builtin_tools(self):
        """Yerleşik tool'ları kaydet."""
        builtin = [
            CalculatorTool(),
            PythonExecutorTool(),
            WebSearchTool(),
            DateTimeTool(),
            WikipediaTool(),
            JSONParserTool(),
            TextAnalyzerTool(),
            UnitConverterTool(),
        ]
        for tool in builtin:
            self.register(tool)
    
    def register(self, tool: BaseTool):
        """Tool kaydet."""
        self._tools[tool.name] = tool
        logger.debug(f"Tool kaydedildi: {tool.name}")
    
    def get(self, name: str) -> Optional[BaseTool]:
        """Tool al."""
        return self._tools.get(name)
    
    def list_tools(self) -> List[Dict]:
        """Tüm tool'ları listele."""
        return [
            {
                'name': tool.name,
                'description': tool.description,
                'parameters': tool.parameters
            }
            for tool in self._tools.values()
        ]
    
    def get_tool_descriptions(self) -> str:
        """Tool açıklamalarını formatla (LLM prompt'u için)."""
        descriptions = []
        for tool in self._tools.values():
            params = ', '.join(tool.parameters.keys())
            descriptions.append(f"- {tool.name}({params}): {tool.description}")
        return "\n".join(descriptions)
    
    async def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """Tool çalıştır."""
        tool = self.get(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                output=f"Tool bulunamadı: {tool_name}",
                tool_name=tool_name
            )
        return await tool.execute(**kwargs)


# Global registry
_tool_registry: Optional[ToolRegistry] = None

def get_tool_registry() -> ToolRegistry:
    """Singleton tool registry."""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
    return _tool_registry


# =============================================================================
# AGENT EXECUTOR
# =============================================================================

class ThoughtStep:
    """ReAct düşünce adımı."""
    
    def __init__(self, thought: str, action: str = None, action_input: str = None,
                 observation: str = None, is_final: bool = False):
        self.thought = thought
        self.action = action
        self.action_input = action_input
        self.observation = observation
        self.is_final = is_final
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict:
        return {
            'thought': self.thought,
            'action': self.action,
            'action_input': self.action_input,
            'observation': self.observation,
            'is_final': self.is_final,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class AgentConfig:
    """Agent konfigürasyonu."""
    max_iterations: int = 5
    max_execution_time: float = 120.0  # saniye (2 dakika)
    verbose: bool = True
    return_intermediate_steps: bool = True


class AgentExecutor:
    """
    ReAct Pattern tabanlı Agent Executor.
    
    ReAct = Reasoning + Acting
    
    Her adımda:
    1. Thought: Durumu değerlendir, ne yapılmalı?
    2. Action: Hangi tool kullanılmalı?
    3. Action Input: Tool'a ne parametre verilmeli?
    4. Observation: Tool sonucu ne?
    5. Repeat veya Final Answer
    """
    
    # ReAct prompt template
    SYSTEM_PROMPT = """Sen yardımcı bir AI asistanısın. Kullanıcı sorularını cevaplamak için çeşitli araçları kullanabilirsin.

Kullanılabilir Araçlar:
{tools}

ÖNEMLİ: 
- Genel bilgi soruları (tarifler, tanımlar, açıklamalar) için ARAÇ KULLANMA, direkt "Final Answer:" ile cevap ver.
- Sadece hesaplama, kod çalıştırma, güncel haber veya spesifik veri gerektiğinde araç kullan.
- HIZLI yanıt ver, gereksiz adım atma.

Yanıt Formatı:
Thought: [Kısa düşünce]
Final Answer: [Cevap]

VEYA araç gerekiyorsa:
Thought: [Neden araç gerekiyor?]
Action: [araç_adı]
Action Input: [parametre]

Kurallar:
1. Tarifler, tanımlar, genel bilgiler → Direkt "Final Answer:" ile cevapla
2. Matematik → calculator kullan
3. Kod çalıştırma → python_executor kullan  
4. Güncel haberler, spesifik veriler → web_search kullan
5. Tarih/saat → datetime kullan

Örnek 1 (Araç gerektirmeyen):
User: Yumurta nasıl haşlanır?
Thought: Bu genel bir tarif sorusu, bilgim var.
Final Answer: Yumurta haşlamak için: 1) Tencereye yumurtaları koyun ve üzerini geçecek kadar su ekleyin. 2) Orta ateşte kaynatın. 3) Rafadan için 6-7 dakika, tam pişmiş için 10-12 dakika bekleyin. 4) Buz suyuna alın ve soyun.

Örnek 2 (Araç gerektiren):
User: 2'nin 10. kuvveti kaç?
Thought: Matematiksel hesaplama, calculator kullanmalıyım.
Action: calculator
Action Input: 2**10
"""
    
    def __init__(self, model_adapter=None, config: AgentConfig = None):
        """
        Args:
            model_adapter: LLM adapter (gemini, huggingface, ollama)
            config: Agent konfigürasyonu
        """
        self.model_adapter = model_adapter
        self.config = config or AgentConfig()
        self.tool_registry = get_tool_registry()
        self.steps: List[ThoughtStep] = []
    
    def _build_prompt(self, user_input: str, history: List[ThoughtStep] = None) -> str:
        """Prompt oluştur."""
        tool_descriptions = self.tool_registry.get_tool_descriptions()
        
        system = self.SYSTEM_PROMPT.format(tools=tool_descriptions)
        
        messages = [system, f"\nUser: {user_input}\n"]
        
        if history:
            for step in history:
                messages.append(f"Thought: {step.thought}")
                if step.action:
                    messages.append(f"Action: {step.action}")
                    messages.append(f"Action Input: {step.action_input}")
                if step.observation:
                    messages.append(f"Observation: {step.observation}")
        
        return "\n".join(messages)
    
    def _parse_response(self, response: str) -> ThoughtStep:
        """LLM yanıtını parse et."""
        thought = ""
        action = None
        action_input = None
        is_final = False
        
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('Thought:'):
                thought = line[8:].strip()
            elif line.startswith('Action:'):
                action = line[7:].strip().lower()
            elif line.startswith('Action Input:'):
                action_input = line[13:].strip()
            elif line.startswith('Final Answer:'):
                thought = line[13:].strip()
                is_final = True
                break
        
        return ThoughtStep(
            thought=thought,
            action=action,
            action_input=action_input,
            is_final=is_final
        )
    
    async def _call_llm(self, prompt: str) -> str:
        """LLM'i çağır."""
        if self.model_adapter is None:
            # Default: Gemini adapter
            try:
                from backend.adapters import gemini
                return await gemini.generate(
                    messages=[{'role': 'user', 'content': prompt}],
                    model_id='gemini-flash'
                )
            except Exception as e:
                logger.error(f"LLM çağrısı başarısız: {e}")
                return f"Final Answer: LLM hatası: {str(e)}"
        else:
            return await self.model_adapter.generate(
                messages=[{'role': 'user', 'content': prompt}]
            )
    
    async def run(self, user_input: str) -> Dict[str, Any]:
        """
        Agent'ı çalıştır.
        
        Args:
            user_input: Kullanıcı sorusu
        
        Returns:
            {
                'output': 'Final cevap',
                'steps': [ThoughtStep, ...],
                'iterations': int,
                'success': bool
            }
        """
        self.steps = []
        start_time = datetime.now()
        
        for iteration in range(self.config.max_iterations):
            # Timeout kontrolü
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > self.config.max_execution_time:
                return {
                    'output': 'Zaman aşımı: İşlem çok uzun sürdü',
                    'steps': [s.to_dict() for s in self.steps],
                    'iterations': iteration,
                    'success': False
                }
            
            # Prompt oluştur
            prompt = self._build_prompt(user_input, self.steps)
            
            if self.config.verbose:
                logger.info(f"[Iteration {iteration + 1}] LLM çağrılıyor...")
            
            # LLM'i çağır
            response = await self._call_llm(prompt)
            
            # Yanıtı parse et
            step = self._parse_response(response)
            
            if self.config.verbose:
                logger.info(f"Thought: {step.thought[:100]}...")
                if step.action:
                    logger.info(f"Action: {step.action} | Input: {step.action_input}")
            
            # Final answer mı?
            if step.is_final:
                self.steps.append(step)
                return {
                    'output': step.thought,
                    'steps': [s.to_dict() for s in self.steps],
                    'iterations': iteration + 1,
                    'success': True
                }
            
            # Tool çalıştır
            if step.action:
                tool_result = await self.tool_registry.execute(
                    step.action,
                    **{list(self.tool_registry.get(step.action).parameters.keys())[0]: step.action_input}
                    if step.action_input else {}
                )
                step.observation = tool_result.output
                
                if self.config.verbose:
                    logger.info(f"Observation: {step.observation[:200]}...")
            
            self.steps.append(step)
        
        # Max iteration'a ulaşıldı
        return {
            'output': 'Maksimum adım sayısına ulaşıldı',
            'steps': [s.to_dict() for s in self.steps],
            'iterations': self.config.max_iterations,
            'success': False
        }
    
    async def stream(self, user_input: str) -> AsyncGenerator[Dict, None]:
        """
        Agent'ı streaming modda çalıştır.
        Her adımda bir event yield eder.
        """
        self.steps = []
        start_time = datetime.now()
        
        yield {'type': 'start', 'input': user_input}
        
        for iteration in range(self.config.max_iterations):
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > self.config.max_execution_time:
                yield {'type': 'error', 'message': 'Zaman aşımı'}
                return
            
            prompt = self._build_prompt(user_input, self.steps)
            
            yield {'type': 'thinking', 'iteration': iteration + 1}
            
            response = await self._call_llm(prompt)
            step = self._parse_response(response)
            
            yield {'type': 'thought', 'thought': step.thought}
            
            if step.is_final:
                self.steps.append(step)
                yield {'type': 'final_answer', 'output': step.thought}
                return
            
            if step.action:
                yield {'type': 'action', 'tool': step.action, 'input': step.action_input}
                
                tool_result = await self.tool_registry.execute(
                    step.action,
                    **{list(self.tool_registry.get(step.action).parameters.keys())[0]: step.action_input}
                    if self.tool_registry.get(step.action) and step.action_input else {}
                )
                step.observation = tool_result.output
                
                yield {'type': 'observation', 'result': step.observation}
            
            self.steps.append(step)
        
        yield {'type': 'error', 'message': 'Maksimum adım sayısına ulaşıldı'}
    
    async def run_async(
        self, 
        query: str, 
        model_id: str = None,
        adapter = None,
        on_thought: Callable[[str], Any] = None,
        on_tool_call: Callable[[str, Dict], Any] = None,
        on_tool_result: Callable[[str, str], Any] = None
    ) -> str:
        """
        Agent'ı WebSocket callback'leri ile çalıştır.
        
        Args:
            query: Kullanıcı sorusu
            model_id: Model ID (opsiyonel)
            adapter: Model adapter (opsiyonel)
            on_thought: Düşünce callback'i
            on_tool_call: Tool çağrısı callback'i
            on_tool_result: Tool sonucu callback'i
        
        Returns:
            Final cevap string'i
        """
        # Geçici olarak adapter'ı ayarla
        original_adapter = self.model_adapter
        if adapter:
            self.model_adapter = adapter
        
        self.steps = []
        start_time = datetime.now()
        final_output = ""
        
        try:
            for iteration in range(self.config.max_iterations):
                # Timeout kontrolü
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > self.config.max_execution_time:
                    return "Zaman aşımı: İşlem çok uzun sürdü"
                
                # Prompt oluştur
                prompt = self._build_prompt(query, self.steps)
                
                # LLM'i çağır - adapter modül olarak geliyor, fonksiyonu doğrudan çağır
                if adapter and hasattr(adapter, 'generate'):
                    response = await adapter.generate(
                        messages=[{'role': 'user', 'content': prompt}],
                        model_id=model_id or 'gemini-flash'
                    )
                else:
                    response = await self._call_llm(prompt)
                
                # Yanıtı parse et
                step = self._parse_response(response)
                
                # Thought callback
                if on_thought and step.thought:
                    try:
                        result = on_thought(step.thought)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        logger.warning(f"on_thought callback error: {e}")
                
                # Final answer mı?
                if step.is_final:
                    self.steps.append(step)
                    final_output = step.thought
                    break
                
                # Tool çalıştır
                if step.action:
                    # Tool call callback
                    if on_tool_call:
                        try:
                            result = on_tool_call(step.action, {'input': step.action_input})
                            if asyncio.iscoroutine(result):
                                await result
                        except Exception as e:
                            logger.warning(f"on_tool_call callback error: {e}")
                    
                    # Tool'u çalıştır
                    tool = self.tool_registry.get(step.action)
                    if tool:
                        param_name = list(tool.parameters.keys())[0] if tool.parameters else 'input'
                        tool_result = await self.tool_registry.execute(
                            step.action,
                            **{param_name: step.action_input} if step.action_input else {}
                        )
                        step.observation = tool_result.output
                        
                        # Tool result callback
                        if on_tool_result:
                            try:
                                result = on_tool_result(step.action, tool_result.output)
                                if asyncio.iscoroutine(result):
                                    await result
                            except Exception as e:
                                logger.warning(f"on_tool_result callback error: {e}")
                    else:
                        step.observation = f"Tool '{step.action}' bulunamadı"
                
                self.steps.append(step)
            
            if not final_output:
                final_output = "Maksimum adım sayısına ulaşıldı, sonuç alınamadı."
            
            return final_output
            
        finally:
            # Adapter'ı geri yükle
            self.model_adapter = original_adapter


# =============================================================================
# SINGLETON & FACTORY
# =============================================================================

_agent_instance: Optional[AgentExecutor] = None

def get_agent(model_adapter=None) -> AgentExecutor:
    """Singleton agent instance."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = AgentExecutor(model_adapter)
    return _agent_instance


def create_agent(model_adapter=None, config: AgentConfig = None) -> AgentExecutor:
    """Yeni agent oluştur (singleton değil)."""
    return AgentExecutor(model_adapter, config)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core classes
    'AgentExecutor',
    'AgentConfig',
    'ThoughtStep',
    
    # Tools
    'BaseTool',
    'ToolResult',
    'ToolRegistry',
    'CalculatorTool',
    'PythonExecutorTool',
    'WebSearchTool',
    'DateTimeTool',
    'WikipediaTool',
    'JSONParserTool',
    'TextAnalyzerTool',
    'UnitConverterTool',
    
    # Factory
    'get_agent',
    'create_agent',
    'get_tool_registry',
]
