"""Order management system."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from loguru import logger


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"


class OrderStatus(Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    """Trading order."""
    id: str
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: float
    price: float | None = None
    stop_price: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)


class OrderManager:
    """Manage order lifecycle."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.logger = logger.bind(module="order_manager")
        self.orders: dict[str, Order] = {}
        self._order_counter = 0

    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: float | None = None,
        stop_price: float | None = None,
    ) -> Order:
        """Create a new order."""
        self._order_counter += 1
        order_id = f"order_{self._order_counter}"

        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
        )

        self.orders[order_id] = order
        self.logger.info(f"Created order: {order_id} {side.value} {quantity} {symbol}")
        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if order_id not in self.orders:
            return False

        order = self.orders[order_id]
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            return False

        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.utcnow()
        self.logger.info(f"Cancelled order: {order_id}")
        return True

    def fill_order(self, order_id: str, filled_price: float, filled_quantity: float | None = None) -> bool:
        """Mark order as filled."""
        if order_id not in self.orders:
            return False

        order = self.orders[order_id]
        order.filled_price = filled_price
        order.filled_quantity = filled_quantity or order.quantity
        order.status = OrderStatus.FILLED
        order.updated_at = datetime.utcnow()
        return True

    def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        """Get all open orders."""
        orders = [o for o in self.orders.values() if o.status in [OrderStatus.PENDING, OrderStatus.OPEN]]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders
