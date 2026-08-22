# Poisson Ritz 多轮 strict-NPU 驻留执行证据

状态：`pass`

- 迭代轮数：250
- CPU loss：`9.948076248168945` → `6.602624893188477`
- NPU 最终 loss：`6.602625370025635`
- CPU/NPU 最大绝对误差：`4.76837158203125e-07`
- H2D：30（仅循环前）
- D2H：5（仅循环后）
- strict-NPU：`true`
- runtime profile：`true`
- runtime target identity：`true`
- 性能：未测量、未发布

## 每轮调用点与累计实际调用

| API | 每轮 | 累计实际调用 |
|---|---:|---:|
| `add` | 11 | 2750 |
| `apply_gradient_descent` | 4 | 1000 |
| `clip` | 3 | 750 |
| `divide` | 14 | 3500 |
| `exp` | 2 | 500 |
| `matmul` | 10 | 2500 |
| `multiply` | 43 | 10750 |
| `negative` | 5 | 1250 |
| `outer` | 14 | 3500 |
| `smooth_l1_loss` | 2 | 500 |
| `sqrt` | 2 | 500 |
| `subtract` | 7 | 1750 |

## 固定身份

- runner SHA-256：`0a60b2202d012aa60d7700cc7c356f2d07f2e14a7b917d74d33fed8ec9627e6d`
- source SHA-256：`1deeeec0e5c32c96c4a33467687ce5181a6c8d2c018c6b81b3dc066e3e8beb1a`
- migrated SHA-256：`49779934abf65eececc5dac074d6ab858ef63ad23999052e531f25069248f195`
- overlay SHA-256：`08625ef3704db848281cab041e1a7da7d69777b1d425a329c6c43cea45ff1b6e`
- capability audit：`asnumpy-0.1.dev106-g21368abf9-poisson-linalg-init4f4e8083-math62e0ba09-direct4479c588-cored3ccb39f-v1+qualification-08625ef3704db848`
