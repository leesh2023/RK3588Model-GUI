#include "mylineedit.h"

#include <QMouseEvent>
#include <QMimeData>

MyLineEdit::MyLineEdit()
    : isMoving{ false }
{}

void MyLineEdit::mousePressEvent(QMouseEvent *event)
{
    Q_UNUSED(event);
    startP = cursor().pos()-window()->pos();
    yuanP = this->pos();
    emit sendSelf(this);   //信号发送该控件地址
    isMoving = true;

    QLineEdit::mousePressEvent(event);
}

//移动事件
void MyLineEdit::mouseMoveEvent(QMouseEvent *event)
{
    Q_UNUSED(event);
    QWidget *parentW = this->parentWidget();

    // 先判断是否可以移动到这里
    QPoint movPoint = yuanP+cursor().pos()-window()->pos()-startP;
    if (movPoint.x() >= 0
        and movPoint.y() >= 0
        and movPoint.x()+this->width() <= parentW->width()
        and movPoint.y()+this->height() <= parentW->height())
    {
        if (isMoving)
            this->move(movPoint);
    } else if (movPoint.x() >= 0
       and movPoint.x()+this->width() <= parentW->width())
    {
        if (isMoving)
            this->move(movPoint.x(), this->y());
    } else if (movPoint.y() >= 0
               and movPoint.y()+this->height() <= parentW->height())
    {
        if (isMoving)
            this->move(this->x(), movPoint.y());
    }

    QLineEdit::mouseMoveEvent(event);
}

void MyLineEdit::dragEnterEvent(QDragEnterEvent *event)
{
    if(event->mimeData()->text() == "Input"
        or event->mimeData()->text() == "Output")
    {
        event->ignore();
    }
    else
    {
        event->accept();
    }
}

void MyLineEdit::dragMoveEvent(QDragMoveEvent *event)
{
    if(event->mimeData()->text() == "Input"
        or event->mimeData()->text() == "Output")
    {
        event->ignore();
    }
    else
    {
        event->accept();
    }
}

void MyLineEdit::dropEvent(QDropEvent *event)
{
    if(event->mimeData()->text() == "Input"
        or event->mimeData()->text() == "Output")
    {
        event->ignore();
    }
    else
    {
        event->accept();
    }
}

//拖拽对象置顶，卡牌积压的时候，拖动的那张卡牌置顶
void MyLineEdit::mouseReleaseEvent(QMouseEvent *event)
{
    Q_UNUSED(event);
    isMoving = false;

    QLineEdit::mouseReleaseEvent(event);
}

