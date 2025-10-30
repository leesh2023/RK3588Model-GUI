#include "mywidgetdraw.h"
#include "QMimeData"

#include <QDragEnterEvent>

MyWidgetDraw::MyWidgetDraw(QWidget *parent)
    : QWidget{parent}
{}


void MyWidgetDraw::dragEnterEvent(QDragEnterEvent *event)
{
    if(event->mimeData()->text() == "Input"
        or event->mimeData()->text() == "Output")
    {
        event->accept();
    }
    else
    {
        event->ignore();
    }
}

void MyWidgetDraw::dragMoveEvent(QDragMoveEvent *event)
{
    if(event->mimeData()->text() == "Input"
        or event->mimeData()->text() == "Output")
    {
        event->accept();
    }
    else
    {
        event->ignore();
    }
}

void MyWidgetDraw::dropEvent(QDropEvent *event)
{
    // 实现拖动控件并创建
    QPoint mouseP = event->position().toPoint();
    if (mouseP.x() >= 0
        and mouseP.y() >= 0
        and mouseP.x() <= this->width()
        and mouseP.y() <= this->height())
    {
        MyLineEdit *mLineEdit = new MyLineEdit();
        mLineEdit->setText(event->mimeData()->text()+":");
        mLineEdit->setParent(this);
        mLineEdit->show();
        mLineEdit->move(mouseP);
        emit sendLineEdit(mLineEdit);
    }
}
