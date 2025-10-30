#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QMouseEvent>
#include "mylineedit.h"
#include "mytoolbutton.h"
#include "mywidgetdraw.h"

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void getObject(MyLineEdit *w);   // 接收MyLineEdit移动信号
    // 视图--工具栏
    void on_action_ToolBar_triggered(bool checked);
    // 视图--控件栏
    void on_action_Object_triggered(bool checked);

private:
    Ui::MainWindow *ui;
    MyLineEdit *curLineEdit;
    std::vector<MyLineEdit*> mInputList;  // 输入
    std::vector<MyLineEdit*> mOutputList;  // 输出
};
#endif // MAINWINDOW_H
