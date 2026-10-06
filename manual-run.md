# 人工測試一場：全本地 CodeTrial

打開 http://127.0.0.1:8095 ，選 **Two Sum**（稿是照它寫的），語言選 Python，
允許麥克風。全程講**英文**（whisper 設成英文、Kokoro 只會英文）。

每句講完停 **3 秒**，面試官才會接話（`GEMINI_SILENCE_MS` 預設 3000）。
思考時沉默超過 3 秒會被當成講完，想停頓就說 "um, let me think"。

## 台詞

| # | 你說 / 你做 | 要觀察 |
|---|---|---|
| 1 | （等面試官開場講完） | 開場有自我介紹、講題目情境、問用什麼語言 |
| 2 | "I'd like to use Python." | 接受並請你複述題目 |
| 3 | "So I get a list of amounts and a target, and I return the two indices whose amounts add up to the target." | 追問限制或邊界 |
| 4 | "Can the amounts be negative?" | 回答可以，沒有多講解法 |
| 5 | **（在面試官講話中途直接插話）** "Sorry, can the same transaction be used twice?" | 面試官立刻停下（打斷），回答你的插話而不是講完原本那句 |
| 6 | "I'm stuck. Can I get a hint?" | 第 1 階提示：問「4 的配對要是多少」之類，**沒有**講 hash map |
| 7 | "The partner of four would be minus four." | 接著引導，不直接給答案 |
| 8 | "Could I get another hint?" | 第 2 階提示：問「走過去時可以記住什麼」 |
| 9 | "Can I get one more hint?" | **不給新提示**，改問你自己會先怎麼做（第 3 階會講出關鍵步驟，CodeTrial 要等你先講出自己的做法才給，這次不計數） |
| 10 | "I'd check every pair with two nested loops. That's O of n squared." | 承認可行，問能不能更快 |
| 11 | "I could keep a dictionary from amount to index as I go." | 請你寫出來 |
| 11b | "Before I code it, could I get one last hint?" | 現在你講過做法了，應該給第 3 階提示 |
| 12 | 在編輯器打下面的**錯誤版本**，按 Run tests | 面試官會對測試失敗有反應 |
| 13 | "One test failed. I insert before I check, so an amount pairs with itself." 然後改成正確版本、再 Run tests | 全過後問複雜度 |
| 14 | "Time is O of n, and space is O of n." | 進到最佳化或下一段 |
| 15 | （行為題）照實回答一個你處理過的技術難題，講 STAR：情境、任務、行動、結果 | 有追問細節 |
| 16 | "I think I'm done." 然後按結束面試 | 等報告；本地最長要 250 秒，頁面 280 秒前不會出現離開按鈕 |

錯誤版本（第 12 步）。函式名稱一定要是頁面 starter 的 `matchDisputedCharge`，否則測試會因找不到函式而無法執行：

```python
def matchDisputedCharge(nums, target):
    seen = {}
    for i, a in enumerate(nums):
        seen[a] = i
        if target - a in seen:
            return [seen[target - a], i]
```

正確版本（第 13 步）：把 `seen[a] = i` 移到 `if` 之後。

## 看報告時檢查

- 有沒有分數、摘要、改進計畫，或顯示 incomplete
- **提示數**應該是 3（第 6、8、11b 步）；第 9 步被擋下不算。看到 4 代表多算，2 代表 11b 沒被記到
- 摘要有沒有寫出 LeetCode 原題名 "Two Sum"（不應該出現）
- 摘要內容和你實際講的對不對得上，有沒有編造
- 行為題有沒有被評到

## 跑的時候另開終端看 log

```bash
tail -f ~/workspace/local-voice-bench/shim.log
```

`heard (...)` 是 whisper 聽到的字；`hint requested: logging it` 是 shim 幫你記提示；
`interrupted` 是打斷生效。報告那段的 log：

```bash
tail -f /tmp/claude-1000/-home-alanhc-workspace/e985b4d2-ba0d-4dab-97c9-53ef00e39149/scratchpad/codetrial-web.log
```

## 跑完記下來（回 jserv 用）

- 整場幾分鐘、報告花多久出來
- 哪幾句被聽錯（對照 `heard`）
- 面試官哪裡不像真人：重複、太長、洩漏答案、該停沒停
- 報告的提示數、分數、有沒有錯誤
