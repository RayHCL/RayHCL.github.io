# RayBlog Hexo 源码

这个分支保存博客的 Hexo 原始工程；`main` 分支保存 `hexo deploy` 生成的静态页面，并由 GitHub Pages 发布。

## 环境

- Node.js 20.19.0 或更高版本
- npm
- Git

首次使用：

```bash
git switch source
nvm use
npm ci
```

## 新建和编辑文章

```bash
npx hexo new "文章标题"
```

文章会创建在 `source/_posts/`。编辑 Markdown 文件中的分类和标签：

```yaml
categories:
  - AI
tags:
  - LangChain
  - Agent
```

## 本地预览

```bash
npx hexo clean
npx hexo server
```

浏览器访问 `http://localhost:4000/`。停止服务时按 `Ctrl+C`。

## 生成静态站点

```bash
npx hexo clean
npx hexo generate
```

生成结果位于 `public/`，该目录不会提交到 `source` 分支。

## 发布

先提交并推送源码：

```bash
git add _config.yml _config.next.yml package.json package-lock.json scaffolds source README.md
git commit -m "Add or update post"
git push origin source
```

再通过 Hexo 生成并部署到 `main`：

```bash
npm run deploy
```

不要直接修改 `main` 分支中的 HTML；重新部署时这些文件会被 Hexo 覆盖。

## 更新依赖

```bash
npm outdated
npm update
npx hexo clean
npx hexo generate
```

依赖升级后，应提交 `package.json` 和 `package-lock.json`。
