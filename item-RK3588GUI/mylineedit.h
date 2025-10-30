#ifndef MYLINEEDIT_H
#define MYLINEEDIT_H

#include <QLineEdit>

class MyLineEdit : public QLineEdit
{
    Q_OBJECT;
public:
    MyLineEdit();

protected:
    void mousePressEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void dragEnterEvent(QDragEnterEvent *event) override;
    void dragMoveEvent(QDragMoveEvent *event) override;
    void dropEvent(QDropEvent *event) override;


signals:
    void sendSelf(MyLineEdit *w);

private:
    bool isMoving;   // 是否在移动
    QPoint startP;   // 移动前距离窗口的位置
    QPoint yuanP;    // 移动前距离屏幕的位置
};

#endif // MYLINEEDIT_H
