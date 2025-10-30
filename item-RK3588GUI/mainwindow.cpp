#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QMimeData>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , curLineEdit{ nullptr }
{
    ui->setupUi(this);

    // 控件初始化设置
    ui->widget_Draw->setAcceptDrops(true);    // 开启识别拖动控件
    ui->action_ToolBar->setChecked(true);     // 初始状态开启
    ui->action_Object->setChecked(true);

    // 信号与槽函数
    connect(ui->widget_Draw, &MyWidgetDraw::sendLineEdit, [=](MyLineEdit *mEdit){
        if (mEdit->text() == "Input:")
            mInputList.push_back(mEdit);
        else if (mEdit->text() == "Output:")
            mOutputList.push_back(mEdit);
        connect(mEdit, &MyLineEdit::sendSelf, this, &MainWindow::getObject);
    });
}

MainWindow::~MainWindow()
{
    delete ui;
}

//获取控件坐标
void MainWindow::getObject(MyLineEdit *w)
{
    curLineEdit = w;
    /*将此小部件提升到父小部件堆栈的顶部*/
    curLineEdit->raise();
}


void MainWindow::on_action_ToolBar_triggered(bool checked)
{
    if (checked)
    {
        ui->toolBar->show();
        // 不改变绘画区控件的位置
        for (auto &edit : mInputList)
        {
            edit->move(edit->x(), edit->y() - ui->toolBar->height());
        }
        for (auto &edit : mOutputList)
        {
            edit->move(edit->x(), edit->y() - ui->toolBar->height());
        }
    } else
    {
        ui->toolBar->hide();
        // 不改变绘画区控件的位置
        for (auto &edit : mInputList)
        {
            edit->move(edit->x(), edit->y() + ui->toolBar->height());
        }
        for (auto &edit : mOutputList)
        {
            edit->move(edit->x(), edit->y() + ui->toolBar->height());
        }
    }
}


void MainWindow::on_action_Object_triggered(bool checked)
{
    if (checked)
    {
        ui->sideBar->show();
        // 不改变绘画区控件的位置
        for (auto &edit : mInputList)
        {
            edit->move(edit->x() - ui->sideBar->width(), edit->y());
        }
        for (auto &edit : mOutputList)
        {
            edit->move(edit->x() - ui->sideBar->width(), edit->y());
        }
    } else
    {
        ui->sideBar->hide();
        // 不改变绘画区控件的位置
        for (auto &edit : mInputList)
        {
            edit->move(edit->x() + ui->sideBar->width(), edit->y());
        }
        for (auto &edit : mOutputList)
        {
            edit->move(edit->x() + ui->sideBar->width(), edit->y());
        }
    }
}

