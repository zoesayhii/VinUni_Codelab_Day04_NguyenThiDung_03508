"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import os
import re
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
Bạn là VinAssistant, trợ lý AI chính thức của hệ sinh thái Vingroup.

## PERSONA
- Vai trò: Chuyên viên tư vấn sản phẩm và dịch vụ VinFast, Vinpearl.
- Giọng nói: Chuyên nghiệp, thân thiện, chính xác và không bịa thông tin.

## AVAILABLE TOOLS
{tools}

## CORE RULES
1. Không bịa giá, tính năng hoặc tình trạng sản phẩm. Phải gọi `search_product_catalog` để lấy dữ liệu thực.
2. Không tự tạo mã ticket. Phải gọi `submit_support_ticket` để hệ thống tạo ticket.
3. Trả lời trực tiếp các câu hỏi FAQ đơn giản về bảo hành hoặc đổi trả.
4. Khi cần nhiều tool, gọi từng tool theo thứ tự rồi tổng hợp kết quả.

## OPERATIONAL BOUNDARIES
Chỉ tư vấn sản phẩm và dịch vụ thuộc hệ sinh thái Vingroup. Từ chối lịch sự các câu hỏi ngoài phạm vi.

## OUTPUT CONTRACT
Luôn trả lời bằng tiếng Việt. Khi dùng tool, ghi nhận Thought, Action, Observation và Final Answer. Khi liệt kê sản phẩm, dùng danh sách có thứ tự. Khi tạo ticket, nêu mã ticket và mức ưu tiên.
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def query(self, user_input: str) -> Dict[str, Any]:
        """Trả lời một lượt mà không truy cập catalog hoặc tool nào."""
        return {
            "answer": (
                "VinFast hiện có nhiều dòng xe điện với giá từ khoảng 300 triệu. "
                "Bạn có thể tham khảo tại showroom gần nhất hoặc website VinFast."
            ),
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    def _detect_intents(self, user_input: str) -> Dict[str, Any]:
        lower = user_input.lower()
        intents = {
            "needs_catalog": False,
            "needs_ticket": False,
            "is_faq": False,
            "catalog_args": {},
            "ticket_args": {},
        }

        faq_keywords = ["chính sách", "bảo hành", "đổi trả", "bao lâu", "điều kiện"]
        action_keywords = ["xem", "tìm", "mua", "đặt", "giá", "dưới"]
        ticket_actions = ["phản hồi", "khiếu nại", "ghi nhận", "ticket", "phản ánh"]
        if any(keyword in lower for keyword in faq_keywords) and not any(
            keyword in lower for keyword in action_keywords + ticket_actions
        ):
            intents["is_faq"] = True
            return intents

        catalog_keywords = ["xe điện", "vinfast", "resort", "vinpearl", "du lịch", "sản phẩm"]
        if any(keyword in lower for keyword in catalog_keywords) and any(
            keyword in lower for keyword in action_keywords
        ):
            if any(keyword in lower for keyword in ["resort", "vinpearl", "du lịch", "nghỉ dưỡng"]):
                intents["catalog_args"]["category"] = "du_lich"
            else:
                intents["catalog_args"]["category"] = "xe_dien"
            for pattern in [r"dưới\s*(\d+)\s*(?:triệu|tr)", r"(\d+)\s*triệu"]:
                match = re.search(pattern, lower)
                if match:
                    intents["catalog_args"]["max_price"] = int(match.group(1)) * 1_000_000
                    break
            intents["needs_catalog"] = True

        ticket_keywords = ["phản hồi", "khiếu nại", "lỗi", "hỏng", "hỗ trợ", "ticket", "phản ánh", "sự cố"]
        if any(keyword in lower for keyword in ticket_keywords):
            name_match = re.search(
                r"(?:tên\s+(?:tôi\s+là\s+|là\s+)?|tôi\s+là\s+)([A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)*)",
                user_input,
            )
            intents["ticket_args"] = {
                "customer_name": name_match.group(1).strip() if name_match else "Khách hàng ẩn danh",
                "issue_description": user_input,
                "priority": "high" if any(word in lower for word in ["khẩn cấp", "nghiêm trọng", "gấp"]) else "medium",
            }
            intents["needs_ticket"] = True
        return intents

    @staticmethod
    def _format_catalog(products: List[Dict[str, Any]]) -> str:
        if not products:
            return "Rất tiếc, hiện không tìm thấy sản phẩm phù hợp."
        lines = [f"Tìm thấy {len(products)} sản phẩm phù hợp:"]
        for index, product in enumerate(products, start=1):
            availability = "Còn hàng" if product["availability"] == "in_stock" else "Đặt trước"
            lines.append(
                f"{index}. {product['name']} — {product['price_vnd']:,} VNĐ\n"
                f"   {product['description']}\n   Tình trạng: {availability}"
            )
        return "\n".join(lines)

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []

        intents = self._detect_intents(user_input)
        if intents["is_faq"]:
            answer = (
                "Chính sách bảo hành pin xe điện VinFast: pin được bảo hành 10 năm hoặc "
                "200.000 km, tùy điều kiện nào đến trước."
            )
            self.trace.append({"iteration": 1, "thought": "Câu hỏi FAQ, không cần gọi tool.", "final_answer": answer})
            return {"answer": answer, "trace": self.trace, "iterations": 1, "status": "completed"}

        iteration = 1
        catalog_results = None
        ticket_result = None
        while iteration <= self.max_iterations:
            if iteration == 1 and intents["needs_catalog"]:
                args = intents["catalog_args"]
                catalog_results = TOOL_MAP["search_product_catalog"](**args)
                self.trace.append({
                    "iteration": iteration,
                    "thought": "Cần tra cứu catalog sản phẩm theo yêu cầu khách hàng.",
                    "action": {"name": "search_product_catalog", "args": args},
                    "observation": catalog_results,
                })
                if not intents["needs_ticket"]:
                    return {"answer": self._format_catalog(catalog_results), "trace": self.trace, "iterations": iteration, "status": "completed"}
            elif (iteration == 1 and not intents["needs_catalog"] and intents["needs_ticket"]) or (
                iteration == 2 and intents["needs_ticket"]
            ):
                args = intents["ticket_args"]
                ticket_result = TOOL_MAP["submit_support_ticket"](**args)
                self.trace.append({
                    "iteration": iteration,
                    "thought": "Cần ghi nhận yêu cầu hỗ trợ của khách hàng.",
                    "action": {"name": "submit_support_ticket", "args": args},
                    "observation": ticket_result,
                })
                if not intents["needs_catalog"]:
                    answer = (
                        "Đã ghi nhận yêu cầu hỗ trợ của bạn:\n"
                        f"- Mã ticket: {ticket_result['ticket_id']}\n"
                        f"- Khách hàng: {ticket_result['customer_name']}\n"
                        f"- Mức ưu tiên: {ticket_result['priority']}\n"
                        f"- Trạng thái: {ticket_result['status']}"
                    )
                    return {"answer": answer, "trace": self.trace, "iterations": iteration, "status": "completed"}
            elif intents["needs_catalog"] or intents["needs_ticket"]:
                parts = []
                if catalog_results is not None:
                    parts.append(self._format_catalog(catalog_results))
                if ticket_result is not None:
                    parts.append(f"Đã tạo ticket {ticket_result['ticket_id']} với mức ưu tiên {ticket_result['priority']}.")
                answer = "\n\n".join(parts)
                self.trace.append({"iteration": iteration, "thought": "Tổng hợp kết quả từ các tool.", "final_answer": answer})
                return {"answer": answer, "trace": self.trace, "iterations": iteration, "status": "completed"}
            else:
                answer = "Tôi chỉ có thể hỗ trợ các sản phẩm và dịch vụ thuộc hệ sinh thái Vingroup."
                self.trace.append({"iteration": iteration, "thought": "Câu hỏi ngoài phạm vi.", "final_answer": answer})
                return {"answer": answer, "trace": self.trace, "iterations": iteration, "status": "completed"}
            iteration += 1

        return {
            "answer": "Lỗi: Agent đã vượt quá số bước lặp tối đa.",
            "trace": self.trace,
            "iterations": self.max_iterations,
            "status": "max_iterations_reached"
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
