# LINE Stamp Creator

LINEスタンプを作成するためのプロジェクトです。

## 概要

このプロジェクトは、LINEスタンプの作成を支援するツールを提供します。
また、Cursor Pro版によりプロジェクトを作成しています。

## 環境構築

1. Python 3.8以降をインストールしてください。
2. プロジェクトディレクトリで仮想環境を作成します。
   ```sh
   python3 -m venv .venv
   ```
3. 仮想環境を有効化します。
   - Linux/macOS:
     ```sh
     source .venv/bin/activate
     ```
   - Windows:
     ```sh
     .venv\Scripts\activate
     ```
4. 依存パッケージをインストールします。
   ```sh
   pip install -r requirements.txt
   ```

## 機能（予定）

- 画像の編集と最適化
- LINEスタンプの規格に準拠した出力
- 一括処理機能 

## GitHub Personal Access Token（PAT）発行・管理手順

### 1. PATの発行方法
1. GitHubにログインし、右上のアイコンから「Settings」を選択。
2. 左メニュー下部の「Developer settings」→「Personal access tokens」→「Tokens (classic)」または「Fine-grained tokens」を選択。
3. 「Generate new token」から新規トークンを作成。
4. アクセス範囲（例：All repositories）や権限（例：Contents: Read and write）を設定。
5. 「Generate token」をクリックし、表示されたトークンを必ずコピーして安全な場所に保存。

![PAT発行画面の例](images/pat_generate_example.png)

### 2. PATの利用方法
- git push/pullなどの操作時、パスワード入力欄にPATを貼り付ける。
- 一度認証すれば、credential.helperの設定により次回以降は自動認証される場合が多い。

### 3. PATの管理・注意点
- トークンは他人に絶対に教えない。
- 万が一漏洩した場合や不要になった場合は、GitHubの「Personal access tokens」画面から削除（Revoke/Delete）する。
- 必要に応じていつでも新しいPATを発行できる。
- cursorはWSL側のgitではなくPowerShell側のgitから操作しているので注意。

--- 