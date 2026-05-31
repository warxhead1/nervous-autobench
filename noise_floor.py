"""Noise-floor measurement for the autobench verdict detector.

Goal: characterize the false-positive rate of the RE detector —
how often does it fire on clean, passing code?

This script:
1. Builds a test corpus of KNOWN-CLEAN code samples across all 14 supported languages
2. Runs each sample through SandboxedExecutor.execute() with verdict detection
3. Measures: how many clean samples incorrectly get CE/RE/TLE/MLE/WA verdicts?
4. Reports per-language false-positive rates
5. Identifies which patterns are most problematic
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from autobench.sandbox import SandboxedExecutor
from autobench.core import Verdict


# ---------------------------------------------------------------------------
# KNOWN-CLEAN test corpus — one sample per category per language
# ---------------------------------------------------------------------------

LANGUAGES = [
    "python", "rust", "go", "javascript", "typescript",
    "java", "c", "cpp", "ruby", "bash", "php", "swift", "kotlin", "zig",
]

# Each entry: (category, description, code)
# We deliberately avoid anything that could trigger RE/CE/TLE/MLE/WA

CORPUS: dict[str, list[dict[str, str]]] = {
    "python": [
        {
            "category": "hello_world",
            "description": "Trivial hello world",
            "code": 'print("Hello, World!")\n',
        },
        {
            "category": "loops",
            "description": "Simple for loop summing numbers",
            "code": "total = 0\nfor i in range(1000):\n    total += i\nprint(total)\n",
        },
        {
            "category": "recursion",
            "description": "Recursive factorial (tail-safe)",
            "code": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\nprint(factorial(10))\n",
        },
        {
            "category": "data_structures",
            "description": "List and dict operations",
            "code": "d = {chr(ord('a') + i): i for i in range(26)}\nlst = [d[k] for k in sorted(d)]\nprint(sum(lst))\n",
        },
        {
            "category": "string_processing",
            "description": "String manipulation",
            "code": 's = "hello world"\nwords = s.split()\nprint(len(words), len(s))\nprint(s.upper()[::-1])\n',
        },
        {
            "category": "file_io",
            "description": "Read stdin and write stdout",
            "code": "import sys\nlines = sys.stdin.readlines()\nprint(len(lines))\n",
        },
    ],
    "rust": [
        {
            "category": "hello_world",
            "description": "Trivial hello world",
            "code": 'fn main() { println!("Hello, World!"); }',
        },
        {
            "category": "loops",
            "description": "Simple for loop summing numbers",
            "code": 'fn main() { let mut total = 0u64; for i in 0..1000 { total += i as u64; } println!("{}", total); }',
        },
        {
            "category": "recursion",
            "description": "Recursive factorial",
            "code": 'fn main() { fn fact(n: u64) -> u64 { if n <= 1 { 1 } else { n * fact(n-1) } } println!("{}", fact(10)); }',
        },
        {
            "category": "data_structures",
            "description": "Vec and HashMap operations",
            "code": 'use std::collections::HashMap; fn main() { let mut m = HashMap::new(); for i in 0..26 { m.insert((b\'a\' + i as u8) as char, i); } let sum: usize = m.values().sum(); println!("{}", sum); }',
        },
        {
            "category": "string_processing",
            "description": "String manipulation",
            "code": 'fn main() { let s = "hello world"; let words: Vec<&str> = s.split_whitespace().collect(); println!("{} {}", words.len(), s.len()); }',
        },
        {
            "category": "file_io",
            "description": "Read stdin and write stdout",
            "code": 'use std::io::{self, Read}; fn main() -> Result<(), Box<dyn std::error::Error>> { let mut input = String::new(); io::stdin().read_to_string(&mut input)?; let lines: Vec<&str> = input.lines().collect(); println!("{}", lines.len()); Ok(()) }',
        },
    ],
    "go": [
        {
            "category": "hello_world",
            "description": "Trivial hello world",
            "code": 'package main\nimport "fmt"\nfunc main() { fmt.Println("Hello, World!") }',
        },
        {
            "category": "loops",
            "description": "Simple for loop summing numbers",
            "code": 'package main\nimport "fmt"\nfunc main() { total := 0; for i := 0; i < 1000; i++ { total += i }; fmt.Println(total) }',
        },
        {
            "category": "recursion",
            "description": "Recursive factorial",
            "code": 'package main\nimport "fmt"\nfunc fact(n int) int { if n <= 1 { return 1 }; return n * fact(n-1) }\nfunc main() { fmt.Println(fact(10)) }',
        },
        {
            "category": "data_structures",
            "description": "Map operations",
            "code": 'package main\nimport "fmt"\nfunc main() { m := make(map[rune]int); for i := 0; i < 26; i++ { m[rune(\'a\'+i)] = i }; sum := 0; for _, v := range m { sum += v }; fmt.Println(sum) }',
        },
        {
            "category": "string_processing",
            "description": "String manipulation",
            "code": 'package main\nimport ("fmt" "strings")\nfunc main() { s := "hello world"; words := strings.Fields(s); fmt.Println(len(words), len(s)) }',
        },
        {
            "category": "file_io",
            "description": "Read stdin and write stdout",
            "code": 'package main\nimport ("bufio" "fmt" "os")\nfunc main() { scanner := bufio.NewScanner(os.Stdin); lines := 0; for scanner.Scan() { lines++ }; fmt.Println(lines) }',
        },
    ],
    "javascript": [
        {
            "category": "hello_world",
            "description": "Trivial hello world",
            "code": 'console.log("Hello, World!");',
        },
        {
            "category": "loops",
            "description": "Simple for loop summing numbers",
            "code": "let total = 0; for (let i = 0; i < 1000; i++) { total += i; } console.log(total);",
        },
        {
            "category": "recursion",
            "description": "Recursive factorial",
            "code": "function fact(n) { return n <= 1 ? 1 : n * fact(n - 1); } console.log(fact(10));",
        },
        {
            "category": "data_structures",
            "description": "Object and array operations",
            "code": "const m = {}; for (let i = 0; i < 26; i++) { m[String.fromCharCode(97 + i)] = i; } const vals = Object.values(m); console.log(vals.reduce((a, b) => a + b, 0));",
        },
        {
            "category": "string_processing",
            "description": "String manipulation",
            "code": 'const s = "hello world"; const words = s.split(" "); console.log(words.length, s.length);',
        },
        {
            "category": "file_io",
            "description": "Read stdin and write stdout",
            "code": "const fs = require('fs'); const input = fs.readFileSync('/dev/stdin', 'utf8'); const lines = input.split('\\n').filter(l => l.length > 0); console.log(lines.length);",
        },
    ],
    "typescript": [
        {
            "category": "hello_world",
            "description": "Trivial hello world",
            "code": 'console.log("Hello, World!");',
        },
        {
            "category": "loops",
            "description": "Simple for loop summing numbers",
            "code": "let total = 0; for (let i = 0; i < 1000; i++) { total += i; } console.log(total);",
        },
        {
            "category": "recursion",
            "description": "Recursive factorial",
            "code": "function fact(n: number): number { return n <= 1 ? 1 : n * fact(n - 1); } console.log(fact(10));",
        },
        {
            "category": "data_structures",
            "description": "Map and array operations",
            "code": "const m = new Map<string, number>(); for (let i = 0; i < 26; i++) { m.set(String.fromCharCode(97 + i), i); } let sum = 0; m.forEach(v => sum += v); console.log(sum);",
        },
        {
            "category": "string_processing",
            "description": "String manipulation",
            "code": 'const s = "hello world"; const words = s.split(" "); console.log(words.length, s.length);',
        },
        {
            "category": "file_io",
            "description": "Read stdin and write stdout",
            "code": "import * as fs from 'fs'; const input = fs.readFileSync('/dev/stdin', 'utf8'); const lines = input.split('\\n').filter(l => l.length > 0); console.log(lines.length);",
        },
    ],
    "java": [
        {
            "category": "hello_world",
            "description": "Trivial hello world",
            "code": 'public class Main { public static void main(String[] args) { System.out.println("Hello, World!"); } }',
        },
        {
            "category": "loops",
            "description": "Simple for loop summing numbers",
            "code": 'public class Main { public static void main(String[] args) { long total = 0; for (int i = 0; i < 1000; i++) { total += i; } System.out.println(total); } }',
        },
        {
            "category": "recursion",
            "description": "Recursive factorial",
            "code": 'public class Main { static long fact(long n) { return n <= 1 ? 1 : n * fact(n - 1); } public static void main(String[] args) { System.out.println(fact(10)); } }',
        },
        {
            "category": "data_structures",
            "description": "HashMap operations",
            "code": 'import java.util.*; public class Main { public static void main(String[] args) { Map<Character, Integer> m = new HashMap<>(); for (int i = 0; i < 26; i++) { m.put((char)(\'a\' + i), i); } int sum = m.values().stream().mapToInt(Integer::intValue).sum(); System.out.println(sum); } }',
        },
        {
            "category": "string_processing",
            "description": "String manipulation",
            "code": 'public class Main { public static void main(String[] args) { String s = "hello world"; String[] words = s.split(" "); System.out.println(words.length + " " + s.length()); } }',
        },
        {
            "category": "file_io",
            "description": "Read stdin and write stdout",
            "code": 'import java.util.*; public class Main { public static void main(String[] args) { Scanner sc = new Scanner(System.in); int lines = 0; while (sc.hasNextLine()) { sc.nextLine(); lines++; } System.out.println(lines); } }',
        },
    ],
    "c": [
        {
            "category": "hello_world",
            "description": "Trivial hello world",
            "code": '#include <stdio.h>\nint main() { printf("Hello, World!\\n"); return 0; }',
        },
        {
            "category": "loops",
            "description": "Simple for loop summing numbers",
            "code": '#include <stdio.h>\nint main() { long total = 0; for (int i = 0; i < 1000; i++) { total += i; } printf("%ld\\n", total); return 0; }',
        },
        {
            "category": "recursion",
            "description": "Recursive factorial",
            "code": '#include <stdio.h>\nlong fact(long n) { return n <= 1 ? 1 : n * fact(n - 1); }\nint main() { printf("%ld\\n", fact(10)); return 0; }',
        },
        {
            "category": "data_structures",
            "description": "Array operations",
            "code": '#include <stdio.h>\nint main() { int arr[26]; long sum = 0; for (int i = 0; i < 26; i++) { arr[i] = i; sum += arr[i]; } printf("%ld\\n", sum); return 0; }',
        },
        {
            "category": "string_processing",
            "description": "String manipulation",
            "code": '#include <stdio.h>\n#include <string.h>\nint main() { const char *s = "hello world"; int words = 1; for (int i = 0; s[i]; i++) { if (s[i] == \' \') words++; } printf("%d %zu\\n", words, strlen(s)); return 0; }',
        },
        {
            "category": "file_io",
            "description": "Read stdin and write stdout",
            "code": '#include <stdio.h>\nint main() { int c, lines = 0; while ((c = getchar()) != EOF) { if (c == \'\\n\') lines++; } printf("%d\\n", lines); return 0; }',
        },
    ],
    "cpp": [
        {
            "category": "hello_world",
            "description": "Trivial hello world",
            "code": '#include <iostream>\nint main() { std::cout << "Hello, World!" << std::endl; return 0; }',
        },
        {
            "category": "loops",
            "description": "Simple for loop summing numbers",
            "code": '#include <iostream>\nint main() { long total = 0; for (int i = 0; i < 1000; i++) { total += i; } std::cout << total << std::endl; return 0; }',
        },
        {
            "category": "recursion",
            "description": "Recursive factorial",
            "code": '#include <iostream>\nlong fact(long n) { return n <= 1 ? 1 : n * fact(n - 1); }\nint main() { std::cout << fact(10) << std::endl; return 0; }',
        },
        {
            "category": "data_structures",
            "description": "Vector and map operations",
            "code": '#include <iostream>\n#include <map>\nint main() { std::map<char, int> m; for (int i = 0; i < 26; i++) { m[\'a\' + i] = i; } long sum = 0; for (auto &p : m) { sum += p.second; } std::cout << sum << std::endl; return 0; }',
        },
        {
            "category": "string_processing",
            "description": "String manipulation",
            "code": '#include <iostream>\n#include <string>\nint main() { std::string s = "hello world"; int words = 1; for (char c : s) { if (c == \' \') words++; } std::cout << words << " " << s.size() << std::endl; return 0; }',
        },
        {
            "category": "file_io",
            "description": "Read stdin and write stdout",
            "code": '#include <iostream>\n#include <string>\nint main() { std::string line; int lines = 0; while (std::getline(std::cin, line)) { lines++; } std::cout << lines << std::endl; return 0; }',
        },
    ],
    "ruby": [
        {
            "category": "hello_world",
            "description": "Trivial hello world",
            "code": 'puts "Hello, World!"',
        },
        {
            "category": "loops",
            "description": "Simple for loop summing numbers",
            "code": "total = 0\nfor i in 0..999\n  total += i\nend\nputs total\n",
        },
        {
            "category": "recursion",
            "description": "Recursive factorial",
            "code": "def fact(n)\n  n <= 1 ? 1 : n * fact(n - 1)\nend\nputs fact(10)\n",
        },
        {
            "category": "data_structures",
            "description": "Hash operations",
            "code": "h = {}\n(0..25).each { |i| h[('a'.ord + i).chr] = i }\nputs h.values.sum\n",
        },
        {
            "category": "string_processing",
            "description": "String manipulation",
            "code": "s = \"hello world\"\nwords = s.split\nputs \"#{words.length} #{s.length}\"\n",
        },
        {
            "category": "file_io",
            "description": "Read stdin and write stdout",
            "code": "lines = 0\nwhile gets\n  lines += 1\nend\nputs lines\n",
        },
    ],
    "bash": [
        {
            "category": "hello_world",
            "description": "Trivial hello world",
            "code": '#!/bin/bash\necho "Hello, World!"',
        },
        {
            "category": "loops",
            "description": "Simple for loop summing numbers",
            "code": '#!/bin/bash\ntotal=0\nfor i in $(seq 0 999); do\n  total=$((total + i))\ndone\necho "$total"',
        },
        {
            "category": "recursion",
            "description": "Recursive factorial (bash has limits)",
            "code": '#!/bin/bash\nfact() { if [ "$1" -le 1 ]; then echo 1; else echo $(( $1 * $(fact $(($1 - 1))) )); fi; }\necho "$(fact 10)"',
        },
        {
            "category": "data_structures",
            "description": "Associative array operations",
            "code": '#!/bin/bash\ndeclare -A m\nfor i in $(seq 0 25); do\n  key=$(printf "\\x$(printf %x $((97 + i)))")\n  m[$key]=$i\ndone\ntotal=0\nfor v in "${m[@]}"; do\n  total=$((total + v))\ndone\necho "$total"',
        },
        {
            "category": "string_processing",
            "description": "String manipulation",
            "code": '#!/bin/bash\ns="hello world"\nwords=$(echo "$s" | wc -w)\nlen=${#s}\necho "$words $len"',
        },
        {
            "category": "file_io",
            "description": "Read stdin and write stdout",
            "code": '#!/bin/bash\nlines=0\nwhile IFS= read -r line; do\n  ((lines++))\ndone\necho "$lines"',
        },
    ],
    "php": [
        {
            "category": "hello_world",
            "description": "Trivial hello world",
            "code": '<?php\necho "Hello, World!\\n";',
        },
        {
            "category": "loops",
            "description": "Simple for loop summing numbers",
            "code": '<?php\n$total = 0;\nfor ($i = 0; $i < 1000; $i++) { $total += $i; }\necho "$total\\n";',
        },
        {
            "category": "recursion",
            "description": "Recursive factorial",
            "code": '<?php\nfunction fact($n) { return $n <= 1 ? 1 : $n * fact($n - 1); }\necho fact(10) . "\\n";',
        },
        {
            "category": "data_structures",
            "description": "Array operations",
            "code": '<?php\n$m = [];\nfor ($i = 0; $i < 26; $i++) { $m[chr(97 + $i)] = $i; }\necho array_sum($m) . "\\n";',
        },
        {
            "category": "string_processing",
            "description": "String manipulation",
            "code": '<?php\n$s = "hello world";\n$words = count(explode(" ", $s));\necho "$words " . strlen($s) . "\\n";',
        },
        {
            "category": "file_io",
            "description": "Read stdin and write stdout",
            "code": '<?php\n$lines = 0;\nwhile ($line = fgets(STDIN)) { $lines++; }\necho "$lines\\n";',
        },
    ],
    "swift": [
        {
            "category": "hello_world",
            "description": "Trivial hello world",
            "code": 'print("Hello, World!")',
        },
        {
            "category": "loops",
            "description": "Simple for loop summing numbers",
            "code": "var total = 0\nfor i in 0..<1000 { total += i }\nprint(total)",
        },
        {
            "category": "recursion",
            "description": "Recursive factorial",
            "code": "func fact(_ n: Int) -> Int { return n <= 1 ? 1 : n * fact(n - 1) }\nprint(fact(10))",
        },
        {
            "category": "data_structures",
            "description": "Dictionary operations",
            "code": "var m: [Character: Int] = [:]\nfor i in 0..<26 { m[Character(UnicodeScalar(97 + i)!)] = i }\nlet sum = m.values.reduce(0, +)\nprint(sum)",
        },
        {
            "category": "string_processing",
            "description": "String manipulation",
            "code": 'let s = "hello world"\nlet words = s.split(separator: " ").count\nprint("\(words) \(s.count)")',
        },
        {
            "category": "file_io",
            "description": "Read stdin and write stdout",
            "code": "var lines = 0\nwhile let line = readLine() { _ = line; lines += 1 }\nprint(lines)",
        },
    ],
    "kotlin": [
        {
            "category": "hello_world",
            "description": "Trivial hello world",
            "code": 'fun main() { println("Hello, World!") }',
        },
        {
            "category": "loops",
            "description": "Simple for loop summing numbers",
            "code": 'fun main() { var total = 0L; for (i in 0..999) { total += i }; println(total) }',
        },
        {
            "category": "recursion",
            "description": "Recursive factorial",
            "code": 'fun main() { fun fact(n: Long): Long = if (n <= 1) 1 else n * fact(n - 1); println(fact(10)) }',
        },
        {
            "category": "data_structures",
            "description": "Map operations",
            "code": 'fun main() { val m = mutableMapOf<Char, Int>(); for (i in 0..25) { m[\'a\' + i] = i }; println(m.values.sum()) }',
        },
        {
            "category": "string_processing",
            "description": "String manipulation",
            "code": 'fun main() { val s = "hello world"; val words = s.split(" ").size; println("$words ${s.length}") }',
        },
        {
            "category": "file_io",
            "description": "Read stdin and write stdout",
            "code": 'fun main() { var lines = 0; while (readLine() != null) { lines++ }; println(lines) }',
        },
    ],
    "zig": [
        {
            "category": "hello_world",
            "description": "Trivial hello world",
            "code": 'pub fn main() void { std.debug.print("Hello, World!\\n", .{}); }',
        },
        {
            "category": "loops",
            "description": "Simple for loop summing numbers",
            "code": 'pub fn main() void { var total: u64 = 0; for (0..1000) |i| { total += i; }; std.debug.print("{d}\\n", .{total}); }',
        },
        {
            "category": "recursion",
            "description": "Recursive factorial",
            "code": 'pub fn main() void { fn fact(n: u64) u64 { return if (n <= 1) 1 else n * fact(n - 1); } std.debug.print("{d}\\n", .{fact(10)}); }',
        },
        {
            "category": "data_structures",
            "description": "Array and slice operations",
            "code": 'pub fn main() void { var sum: u64 = 0; for (0..26) |i| { sum += i; }; std.debug.print("{d}\\n", .{sum}); }',
        },
        {
            "category": "string_processing",
            "description": "String manipulation",
            "code": 'pub fn main() void { const s = "hello world"; var words: u32 = 1; for (s) |c| { if (c == \' \') words += 1; }; std.debug.print("{d} {d}\\n", .{words, s.len}); }',
        },
        {
            "category": "file_io",
            "description": "Read stdin and write stdout",
            "code": 'pub fn main() !void { const stdin = std.io.getStdIn().reader(); const stdout = std.io.getStdOut().writer(); var buf: [1024]u8 = undefined; var lines: u32 = 0; while (try stdin.readUntilDelimiterOrEof(\'\\n\', &buf)) |line| { _ = line; lines += 1; } try stdout.print("{d}\\n", .{lines}); }',
        },
    ],
}


@dataclass
class NoiseFloorResult:
    """Result of running one sample."""
    language: str
    category: str
    description: str
    expected_verdict: Verdict
    actual_verdict: Verdict
    is_false_positive: bool
    latency_ms: float
    stderr: str = ""
    stdout: str = ""


@dataclass
class LanguageStats:
    """Per-language aggregated statistics."""
    language: str
    total_samples: int = 0
    false_positives: int = 0
    ce_count: int = 0
    re_count: int = 0
    tle_count: int = 0
    mle_count: int = 0
    wa_count: int = 0
    ok_count: int = 0
    results: list[NoiseFloorResult] = field(default_factory=list)

    @property
    def false_positive_rate(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.false_positives / self.total_samples


@dataclass
class NoiseFloorReport:
    """Full noise-floor report."""
    total_samples: int = 0
    total_false_positives: int = 0
    overall_false_positive_rate: float = 0.0
    language_stats: dict[str, LanguageStats] = field(default_factory=dict)
    problematic_patterns: list[dict[str, Any]] = field(default_factory=list)


def run_noise_floor_measurement(
    executor: SandboxedExecutor | None = None,
    verbose: bool = False,
) -> NoiseFloorReport:
    """Run the full noise-floor measurement.

    Args:
        executor: SandboxedExecutor instance to use.
        verbose: If True, print each sample result.

    Returns:
        NoiseFloorReport with per-language and overall statistics.
    """
    executor = executor or SandboxedExecutor(max_memory_mb=512)

    report = NoiseFloorReport()
    constraints = {"max_time_seconds": 30, "max_memory_mb": 512}

    for lang in LANGUAGES:
        samples = CORPUS.get(lang, [])
        lang_stats = LanguageStats(language=lang)

        for sample in samples:
            result = _run_sample(executor, lang, sample, constraints, verbose)
            lang_stats.results.append(result)
            lang_stats.total_samples += 1

            if result.actual_verdict == Verdict.OK:
                lang_stats.ok_count += 1
            elif result.actual_verdict == Verdict.CE:
                lang_stats.ce_count += 1
            elif result.actual_verdict == Verdict.RE:
                lang_stats.re_count += 1
            elif result.actual_verdict == Verdict.TLE:
                lang_stats.tle_count += 1
            elif result.actual_verdict == Verdict.MLE:
                lang_stats.mle_count += 1
            elif result.actual_verdict == Verdict.WA:
                lang_stats.wa_count += 1

            if result.is_false_positive:
                lang_stats.false_positives += 1

        report.language_stats[lang] = lang_stats
        report.total_samples += lang_stats.total_samples
        report.total_false_positives += lang_stats.false_positives

    report.overall_false_positive_rate = (
        report.total_false_positives / report.total_samples
        if report.total_samples > 0
        else 0.0
    )

    # Analyze problematic patterns
    report.problematic_patterns = _analyze_problematic_patterns(report)

    return report


def _run_sample(
    executor: SandboxedExecutor,
    language: str,
    sample: dict[str, str],
    constraints: dict[str, Any],
    verbose: bool,
) -> NoiseFloorResult:
    """Run a single sample and return the result."""
    code = sample["code"]
    category = sample["category"]
    description = sample["description"]

    result = executor.execute(
        code=code,
        language=language,
        constraints=constraints,
    )

    actual_verdict = result.verdict
    # All samples are KNOWN-CLEAN so expected verdict is always OK
    expected_verdict = Verdict.OK
    is_fp = actual_verdict != expected_verdict

    noise_result = NoiseFloorResult(
        language=language,
        category=category,
        description=description,
        expected_verdict=expected_verdict,
        actual_verdict=actual_verdict,
        is_false_positive=is_fp,
        latency_ms=result.latency_ms,
        stderr=result.stderr[:500] if result.stderr else "",
        stdout=result.stdout[:200] if result.stdout else "",
    )

    if verbose and is_fp:
        print(f"  [FP] {language}/{category}: expected OK, got {actual_verdict.value}")
        if result.stderr:
            print(f"       stderr: {result.stderr[:200].strip()}")

    return noise_result


def _analyze_problematic_patterns(report: NoiseFloorReport) -> list[dict[str, Any]]:
    """Identify which verdict categories and languages have the most FPs."""
    patterns = []

    # Per-language breakdown
    for lang, stats in report.language_stats.items():
        if stats.false_positives > 0:
            patterns.append({
                "type": "language",
                "language": lang,
                "false_positives": stats.false_positives,
                "total": stats.total_samples,
                "fp_rate": stats.false_positive_rate,
                "verdicts": {
                    "CE": stats.ce_count,
                    "RE": stats.re_count,
                    "TLE": stats.tle_count,
                    "MLE": stats.mle_count,
                    "WA": stats.wa_count,
                },
            })

    # Per-category breakdown across all languages
    category_fps: dict[str, dict[str, int]] = {}
    for lang, stats in report.language_stats.items():
        for result in stats.results:
            if result.is_false_positive:
                if result.category not in category_fps:
                    category_fps[result.category] = {"total": 0}
                category_fps[result.category]["total"] += 1

    for cat, counts in category_fps.items():
        patterns.append({
            "type": "category",
            "category": cat,
            "false_positives": counts["total"],
        })

    # Sort by false positive count descending
    patterns.sort(key=lambda p: p.get("false_positives", 0), reverse=True)

    return patterns


def print_report(report: NoiseFloorReport) -> None:
    """Print a human-readable noise-floor report."""
    print("=" * 80)
    print("NOISE FLOOR CHARACTERIZATION REPORT")
    print("=" * 80)
    print()
    print(f"Total samples:     {report.total_samples}")
    print(f"False positives:  {report.total_false_positives}")
    print(f"Overall FP rate:  {report.overall_false_positive_rate:.2%}")
    print()
    print("-" * 80)
    print("PER-LANGUAGE BREAKDOWN")
    print("-" * 80)
    print(f"{'Language':<14} {'Total':>6} {'FP':>4} {'FP%':>6}  CE   RE  TLE  MLE   WA   OK")
    print("-" * 80)

    for lang in sorted(report.language_stats.keys()):
        stats = report.language_stats[lang]
        fp_pct = stats.false_positive_rate * 100
        print(
            f"{lang:<14} {stats.total_samples:>6} {stats.false_positives:>4} "
            f"{fp_pct:>5.1f}%  "
            f"{stats.ce_count:>3} {stats.re_count:>3} {stats.tle_count:>4} "
            f"{stats.mle_count:>4} {stats.wa_count:>4} {stats.ok_count:>3}"
        )

    print("-" * 80)

    # Per-category breakdown
    print()
    print("-" * 80)
    print("PER-CATEGORY BREAKDOWN (false positives only)")
    print("-" * 80)

    cat_stats: dict[str, dict[str, int]] = {}
    for lang, stats in report.language_stats.items():
        for result in stats.results:
            cat = result.category
            if cat not in cat_stats:
                cat_stats[cat] = {"total": 0, "fps": 0}
            cat_stats[cat]["total"] += 1
            if result.is_false_positive:
                cat_stats[cat]["fps"] += 1

    for cat in sorted(cat_stats.keys()):
        counts = cat_stats[cat]
        fp_rate = counts["fps"] / counts["total"] if counts["total"] > 0 else 0
        print(f"  {cat:<20}  total={counts['total']:>3}  fps={counts['fps']:>3}  rate={fp_rate:.1%}")

    print("-" * 80)

    # Detailed FP listing
    print()
    print("FALSE POSITIVE DETAILS")
    print("-" * 80)

    all_fps = []
    for lang, stats in report.language_stats.items():
        for result in stats.results:
            if result.is_false_positive:
                all_fps.append(result)

    if not all_fps:
        print("  No false positives detected.")
    else:
        for fp in all_fps:
            print(f"  [{fp.actual_verdict.value}] {fp.language}/{fp.category}")
            print(f"    Description: {fp.description}")
            if fp.stderr:
                for line in fp.stderr.splitlines()[:3]:
                    print(f"    stderr: {line[:80]}")
            print()

    print("-" * 80)
    print()


def main() -> None:
    """Run noise floor measurement and print report."""
    import argparse
    parser = argparse.ArgumentParser(description="Noise-floor measurement for verdict detector")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print each sample result")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument(
        "--sandbox-type",
        default="subprocess",
        choices=["subprocess", "gvisor"],
        help="Sandbox type (default: subprocess)",
    )
    args = parser.parse_args()

    executor = SandboxedExecutor(sandbox_type=args.sandbox_type)
    report = run_noise_floor_measurement(executor, verbose=args.verbose)

    if args.json:
        # Serialize to dict for JSON output
        def _result_to_dict(r: NoiseFloorResult) -> dict:
            return {
                "language": r.language,
                "category": r.category,
                "description": r.description,
                "expected_verdict": r.expected_verdict.value,
                "actual_verdict": r.actual_verdict.value,
                "is_false_positive": r.is_false_positive,
                "latency_ms": r.latency_ms,
                "stderr": r.stderr,
                "stdout": r.stdout,
            }

        def _stats_to_dict(s: LanguageStats) -> dict:
            return {
                "total_samples": s.total_samples,
                "false_positives": s.false_positives,
                "false_positive_rate": s.false_positive_rate,
                "ce_count": s.ce_count,
                "re_count": s.re_count,
                "tle_count": s.tle_count,
                "mle_count": s.mle_count,
                "wa_count": s.wa_count,
                "ok_count": s.ok_count,
                "results": [_result_to_dict(r) for r in s.results],
            }

        output = {
            "total_samples": report.total_samples,
            "total_false_positives": report.total_false_positives,
            "overall_false_positive_rate": report.overall_false_positive_rate,
            "language_stats": {k: _stats_to_dict(v) for k, v in report.language_stats.items()},
            "problematic_patterns": report.problematic_patterns,
        }
        print(json.dumps(output, indent=2))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
