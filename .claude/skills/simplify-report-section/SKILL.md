---
name: simplify-report-section
description: Rewrite a section of the FFI26 LaTeX report (report/main_revised.tex) so it is as detailed as the matching section in report/main.tex but still simple, and so every proposal is justified by results earlier in the report. Use when the user asks to "sửa phần X cho chi tiết như main.tex nhưng không overcomplicate", simplify a report section, or explain why a next step follows from the results.
---

# Simplify a report section (FFI26)

Goal: a section that reads simply, but leaves out nothing the reader needs to understand *what* is proposed and *why*.

## Steps

1. **Find both versions.** `grep -n` the section heading in `report/main_revised.tex` (target) and `report/main.tex` (detailed source). Read both versions of the section, plus the result sections the target relies on (Kết quả, Giả thuyết nào đúng?, Chuỗi phiên bản...).
2. **Port the needed content, not the clutter.** Copy from main.tex:
   - the mechanism in one sentence (what the method changes or learns),
   - an expected outcome, labelled "Kỳ vọng (chưa kiểm chứng)",
   - its trade-off or limit,
   - the line saying the report has not tested it, if that applies.
   Leave out file paths, run IDs, uncorrected p-values and long lists. Detail like that belongs in notes, not the .tex.
3. **Tie the proposal to earlier results.** Write one sentence such as "Đề xuất này xuất phát từ ... quan sát ở kết quả ...". Follow it with a short `itemize` where each bullet has:
   - a **bold claim** naming the finding,
   - one sentence of evidence, taken from a table or paragraph already in the report (reuse only numbers that are already there),
   - one clause on how the proposal deals with it.
4. **Keep it lean.** Two or three bullets at most. Drop low-level technical bullets (e.g. bf16 rounding, update magnitude) unless they are the main reason. The user cut one of these from the DSRL paragraph.
5. **Be honest about limits.** Finish by saying what the proposal probably *cannot* fix, tied to a hypothesis that failed (e.g. H3). Do not present reasoning as a measured result. If a bullet is inference, tell the user in chat so they can decide whether to soften it ("Khả năng cao là…").
6. **Compile and check.** In `report/`, run `latexmk -xelatex -interaction=nonstopmode main_revised.tex`, then `grep -c "^!" main_revised.log`. It should print 0.

## Style

- The report text is in Vietnamese; chat with the user in English.
- Write in the voice of `report/smolvla_rl_A3_A5.tex`: short sentences, few numbers.
- Do not commit. Leave edits in the working tree.

## Example (DSRL paragraph, final accepted form)

```latex
\paragraph{Hướng tiếp theo: DSRL} (Wagenmaker và cộng sự, 2025). Giữ nguyên SmolVLA, học một policy nhỏ chọn nhiễu ban đầu
$\varepsilon$ của flow matching, nên không cần gradient qua VLA. Đề xuất này xuất phát từ ba quan sát ở kết quả FPO:
\begin{itemize}
\item \textbf{Sửa trọng số dễ gây quên.} v1--v3 đều làm giảm 8 task còn lại; ... DSRL không đụng vào trọng số SmolVLA nên tránh được vấn đề này từ đầu.
\item \textbf{Phần FPO sửa được là hành vi SmolVLA đã có sẵn.} Advantage được tính so với 8 lần chạy cùng cảnh, ... Chọn nhiễu ban đầu là cách trực tiếp hơn để luôn lấy ra những hành vi tốt đó.
\end{itemize}
Kỳ vọng (chưa kiểm chứng): ... Đổi lại, ... có thể cũng không sửa được lỗi gắp ở task 0 (H3). Báo cáo này chưa đánh giá DSRL.
```
